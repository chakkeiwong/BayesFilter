# Recovery campaign: repair the inherited step-size grid

The initial fresh factor and strict charts both produced `no_viable_candidate`:
all eight inherited `(epsilon, L)` pairs failed all-chain movement. Larger
steps have very low acceptance; even epsilon 0.055 leaves at least one chain
unmoved. This rejects the tested settings, not either backend or the research
direction. Both committed charts are intact. Worker lifetimes consumed
551.4929145380156 aggregate seconds, leaving 35,448.50708546198 of the owner's
36,000 seconds. No runtime or recovery canary result has yet been obtained.

Keep those exact charts, target, four-chain bank, GPU assignments, precision,
and public measured-grid tuner. Test eight new pairs: epsilon
`(0.0275, 0.01375, 0.006875, 0.0034375)` crossed with the unchanged `L=(3,8)`.
These are a geometric step-size contraction of the failed grid's minimum,
derived as `0.055 / 2**j`, j=1..4. They are repair hypotheses, not defaults.
Use new tuner/calibration/heldout seeds by a deterministic +10,000 offset to
the existing tuning roots, disjoint from the original run's offsets. Do not
retrain charts, change initialization, repeatedly select new seeds, or weaken
movement/finite/status/heldout checks to manufacture a pass.

The primary criterion remains a public verified handoff followed by exact
interrupted/uninterrupted canary equality and two complete trace-valid runtime
measurements. Candidate acceptance is explanatory and a repair trigger, not a
promotion criterion or evidence of mixing. No statistically supported ranking
is sought. If the smaller grid still has no viable setting, inspect its full
diagnostics before another explicitly documented budgeted repair.

Only the diagnostic launcher changes. Record its exact source revision migration
in the same campaign, retain the original campaign-start and failed attempt,
and preserve all historical ledger entries/debits. The chart store retains its
original source identity; fresh repaired tuning is stored under a distinct key
and verified against unchanged numerical-library sources. The active ledger
source hash is updated only with zero outstanding reservations. Its total
allocation remains 36,000 seconds. This is ordinary experiment revision
provenance, not a new approval token or a budget reset.

The implementation audit found that the earlier successful grid was only a
warm start for these new charts. Its failure is precisely why fresh scope
tuning is required. No target, method, convergence criterion, or resource
policy changes. The original 1,200-second reference-worker cap applies to
this retry; released reservations and the original repair reserve cover it.

Concurrent-work audit: another workspace task added
`fixed_transport_candidate_selection.py` and lazy exports for it in
`bayesfilter/inference/__init__.py` after the original source snapshot. The
q=20 public fixed-transport tuner and mechanics module do not import this new
module. The inspected initializer diff only adds its distinct exports; it
does not replace the public tuner or chart implementation. Preserve those
unrelated changes, record their hashes in the migration, and issue fresh
tuning under the current initializer hash. All numerical files actually used
by the prior chart builder remain unchanged. No previous handoff exists to
silently upgrade; the prior failed grid remains failed evidence.

## Setup-timeout repair

The repaired factor grid passed and committed its verified handoff; both
eight-transition reference calls passed complete health and single-trace XLA
checks. Strict completed 23 of the tuner's internal calls before its external
1,170-second TERM boundary. This is an infrastructure/time-allocation failure,
not a failed heldout result or a candidate rejection. The 1,200-second envelope
underestimated selection cost when more grid candidates survived. Factor used
838.0603902129806 seconds; strict used 1,170.2638360059937 seconds. Campaign debit
is now 2,559.81714075699 seconds, with 33,440.18285924301 left.

Retry the reference wave with **2,000 seconds per worker**, including the same
30-second cleanup grace. This convenience ceiling is derived by rounding the
observed 1,170-second incomplete run upward to leave time for remaining
selection/heldout and two short canary calls; it is below the owner's 14,400-
second arm cap. Factor replays its committed setup and chunks. Strict reuses
the committed chart and repeats its unfinished tuning stage with the same
declared repaired grid/seeds. Individual tuning calls were logged but are not
independently reusable checkpoints; no completed tuning result is fabricated
from them. This is an explicit recovery-granularity limitation, not permission
to lose a committed chart, tuning result, or sampling chunk.

Use the existing coordinator's `_wave(..., "reference", 2000.0, ...)` with the
recorded migration and unchanged source. The exact parent command and new
child jobs are archived. Resume the ordinary canary command afterward; it
skips the now-completed reference wave. Total authority remains 36,000 seconds,
all unsuccessful attempts remain charged, and no scientific criterion changes.

## Queue and successful-sibling repair

Both reference arms passed. Both interrupt workers were SIGKILLed inside their
second compiled call after the first chunk was durably archived and health
checked. Factor resumed successfully. Strict did not launch on the first resume
wave: the launcher asked the policy selector for only one remaining GPU, got
the free factor GPU, then rejected it because the strict comparator is bound
to GPU 0. This was a queue bug, not genuine proof GPU 0 was unavailable.

Repair selection to inspect both eligible non-display devices before matching
the pending arm's recorded UUID. Preserve successful sibling manifests across
failed waves; do not repeat their work or debit them again. Only orchestration
changes: the chart, tuner, shared kernel, chunk seeds, and diagnostics are
unchanged. Record source migration r2 with the exact pre-repair stream source
identity preserved for replay and the actual launcher revision in new job
manifests. Test pending-arm selection and successful-result preservation on CPU
before retry. Current campaign spending is 3,997.135252105043 seconds; no active
reservations remain. The original 400-second resume-worker cap still applies.

One secondary coordinator console log (`preflight/canary-repair-r1-launch.log`)
was inadvertently reused by a subsequent shell redirect. The earlier failed
worker logs, immutable job/result/summary files, timings, and ledger debits are
preserved; no scientific tensor or checkpoint evidence was overwritten. All
subsequent shell logs use fresh names.
