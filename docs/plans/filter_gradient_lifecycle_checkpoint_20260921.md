# Native sequential lifecycle checkpoint

Status: qualification in progress on
`repair/filter-gradient-xla-validation-20260918`; no terminal admission or merge.
Baseline for this increment: `cfbc32d2`. The public outer loop has not yet been
replaced. The broader master findings, numerical discrepancies and terminal
repeated comparisons remain open.

The new TensorFlow modules enclose the existing refinement/terminal recurrence,
actual fresh score fitting and ordered factor proposals. The recurrence retains
the original asymmetric budget checks, exact incumbent promotion, terminal
retry seeds, radius/stall updates, rejection precedence and final-fit behavior.
It preserves the original final fit's lack of recentering. Histories use fixed
tensor storage; cumulative evaluation counts are int64, including checks around
the 2^31 boundary. `terminal_called` distinguishes completed fitting from a stop
before fitting without changing the numerical decisions.

Actual callback tests use the exact original public-tail source from Git and
load its numerical dependencies recursively into an isolated namespace. They
compare all endpoint fields, history rows, movement diagnostics, progress-event
contents/order, target-row counts and row order. Final mass/report construction
is outside the numerical lifecycle under test. Diagnostic progress comparison
does not establish identical live callback timing.

| Evidence | Result |
| --- | --- |
| Lifecycle decision mechanics, CPU/GPU | 14 cases each pass; latest 01748/01749 includes terminal_called observations. |
| Actual terminal provider, CPU/GPU | Seven cases each pass in 01726/01727. |
| Actual refinement provider, CPU/GPU | Six cases each pass in 01732--01736, including real second-factor fitting. |
| Actual enclosing lifecycle after the gather repair, CPU/GPU | Eight cases each pass in 01757--01764; original settings and 1e-10 comparisons retained. |
| Changed-input HLO and resources, CPU/GPU | D3/D5 pass in 01753--01755: seven/eight runtime operands, one trace, matching HLO, execution after Python cache eviction, release of one/two actual validity resources. |
| Shared selected-row consumer checks | 20 selection cases (01766), including the selected-row pullback, and 17 complete score-fit cases (01767) pass. |
| Locator and public sequential/block consumers | 33 locator cases (01768), 40 original sequential cases (01769) and 43 block cases (01770) pass. |
| Policy/controller checks | All 67 pass in 01765; the second matrix reuses that unchanged-source evidence. |

The enclosing test exposed input-dependent XLA specialization that earlier
single-trace checks missed. Run 01750 retained only four of seven lifecycle
operands; initial center, scale and target precision were embedded constants.
Run 01751 localized it to the terminal and refinement helpers, which retained
one of six and two of nine operands respectively. Six data-dependent scalar
winner reads in `sequential_selection_tf.py` and `sequential_score_fit_tf.py`
were replaced with equivalent `tf.gather` calls. Run 01752 restored all six/nine
dependency operands; the complete changed-input and consumer checks above
qualify that execution repair. No winning index, formula, seed, optimizer
setting, derivative rule or tolerance was changed.

Preserved harness failures are separate from that runtime defect. Run 01722
used int32 observer resources that TensorFlow placed on CPU; only test observers
were moved to int64. Run 01738 constructed a diagnostic range with dynamic
endpoints; static range extent plus the same dynamic offset repaired its XLA
shape requirement. Neither failure justifies an eager fallback.

The guard covers 198 sources with the same 1,276 exact exceptions. The three new
runtime modules have no exceptions. This is explicitly partial coverage;
broader controller and repository-wide terminal gates remain open.

The first six fresh-process costs (01771--01776) compare frozen cfbc32d2,
explicit graph and current XLA at D3/search4 and D5/search32. Full fitting allows
200 optimizer iterations, and every arm includes complete endpoint/progress
records, twenty warm calls and two changed-input/repeat pairs. Frozen/current
XLA records are exactly equal. Strict graph/XLA passes D3 but fails 19 D5 fields
at unchanged atol=rtol=1e-10, including terminal precision/covariance and the
factor Jacobian condition. Specific source attribution is still pending.

The original diagnostic materializer dispatched a tensor slice for every saved
history field. Copying each completed history column once preserves all eight
actual lifecycle records on both CPU/GPU (01777--01784); 67 policy/controller
checks pass in 01785. The renewed costs (01786--01791) retain exact frozen/XLA
records and the same 19 failed D5 mode fields. No numerical source changed.

| Descriptive measurement | D3 before / XLA | D5 before / XLA |
| --- | --- | --- |
| Cold complete call including build | 22.156 / 25.805 s | 64.534 / 76.266 s |
| Warm complete median | 60.330 / 70.028 ms | 175.411 / 195.831 ms |
| XLA numerical / report medians | 33.049 / 36.693 ms | 146.783 / 48.603 ms |
| Additional host peak | 492.066 MiB | 1320.527 MiB |
| GPU allocator peak | 94,976 / 190,464 bytes | 230,912 / 417,280 bytes |

The original warm complete XLA medians were 98.980/233.019 ms, including
65.975/85.947 ms reporting. Bulk transport reduces that descriptive overhead;
renewed warm before/after increases are 16.1%/11.6%, below the 20% trigger.
Cold host overhead remains unresolved. D3's revised report transport increases
its small GPU peak just beyond 2x, which is also an open trigger. Graph/XLA D5
cold cost crosses 2x. Twenty warm XLA calls add 80 KiB host RSS at both extents;
two changed inputs add approximately 148/212 KiB versus 50.36/41.17 MiB in the
frozen arm. These bounded observations do not prove general leak freedom.

Both comparisons and the exact analysis script are archived at the campaign
root: lifecycle-memory-comparison-01776.json, -01791.json and
analyze_lifecycle_memory_20260921.py. They are single-process explanatory costs,
not terminal repeated evidence or public integration qualification.

Stage profiles 01792/01793 pass. Logical raw outputs are 3,035/6,875 bytes;
outer graphs have 10,900/26,225 nodes and optimized HLO is approximately 8.6/18.5
MB. RSS grows from roughly 1.05 GB after preparation to 1.30/1.76 GB after outer
tracing and 2.28/4.36 GB immediately after raw execution. Reporting follows that
increase. The profile records /proc map counts and allocator values throughout.
This places the major observed increase before report construction, although
it does not attribute every allocation to a particular compiler data structure.

The /proc VmHWM field decreases by about 31--39 MiB between raw execution and
reporting. Preserve this anomaly; status snapshots alone cannot establish an
exact process maximum. A planned measurement-only cross-check adds getrusage
peak RSS and smaps_rollup residency. No OS or allocator setting changes.

Terminal localization 01794 reproduces the same strict matrix failures with
both frozen cfbc32d2 and current source at all three saved centers/scales/seeds.
At identical pre-eigensystem precision, raw XLA residual is 1.788e-10, compared
with <=8.882e-16 for CPU/GPU graph and the repository's existing refined helper.
That attributes the terminal precision gap to premature eigensolver convergence;
it does not qualify a runtime repair. The factor condition diagnostic remains
under investigation. Its first attempt (01795) incorrectly requested raw
optimizer coordinates from the deliberately compact OptimizerSummary. The retry
uses the identical full dependency and first checks every retained field/count.

Public progress migration also remains open. The
[consumer review](filter_gradient_lifecycle_progress_review_20260921.md) identifies
one external callback-driven wall cap and one semantic-progress watchdog.
Compiled outer progress needs explicit buffered-delivery semantics and an
independent process deadline before those consumers can be admitted.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain native candidate; continue bounded investigations | Complete frozen/XLA records and all affected consumers pass | Strict D5 graph/XLA failures, cold host growth and D3 GPU ratio remain open | Numerical source attribution, compilation memory and external progress timing | Profile stages, localize failed fields, then qualify public integration | No complete repair, terminal performance, external DZ5 full-transition-block or merge admission |

Primary-agent review: the strongest alternative explanation for apparent
compiled reuse was hidden value specialization; explicit runtime operands and
changed-input HLO disprove it in D3/D5 after the repair. The weakest evidence is
scope: small mechanics/geometry cases do not qualify arbitrary capacities,
long-lived CPU executable retention, external targets or watchdogs. Python
factory eviction is not native compiler-cache eviction. A failed remaining
consumer or cost comparison triggers further repair, not a relaxed gate.

Final review through 01799: the four attribution jobs finish after the recorded
01795 harness repair. At identical raw factor coordinates, frozen/current and
graph/XLA Jacobian conditions agree within 3.6e-14. The enclosing 9.55e-9 gap
comes from fitted-state differences of up to 1.019e-10, leaving the upstream
fitting/preparation comparison open. Terminal defects are specifically tied to
the eigensolver; they are not dismissed merely as inherited rounding.

The memory cross-check repeats both extents (01797/01798), retaining exact cost
records. smaps_rollup confirms raw RSS at 2.279/4.366 GB, while getrusage peak is
lower by 37.9/27.9 MB. The archived stage disposition preserves every reading.
Most extra memory is resident before reporting; exact peak accounting and final
repeated resource admission remain unproven. No full-device preallocation occurs.

All 67 policy/controller checks pass in 01799, and focused Ruff/whitespace pass.
No worker is active. Charges: CPU 41,166.158309666 / 115,200 and GPU
36,678.672169596 / 187,200 seconds. Remote fetch shows the repair branch has no
divergence from cfbc32d2. Commit/push this partial qualification checkpoint.
Primary-agent review only; no independent reviewer was launched. Strongest
remaining alternative explanations are input/preparation rounding for the factor
fit and duplicate graph/compiler structures for cold host cost. The next checks
must isolate those and the actual public/external call chain. No merge admission.
