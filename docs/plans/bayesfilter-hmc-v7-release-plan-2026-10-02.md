# Replicated-trial HMC tuning: scoped release execution

Current status, October 5: **P4 independent confirmation is running; release
remains pending**. The frozen [execution design](artifacts/hmc-v7-release-2026-10-02/confirmation-execution-2026-10-05.md)
passed source, configuration, seed, cost and storage checks. It contains 96
independent searches, interleaved 32 per original family. The persistent service
`bayesfilter-hmc-v7-confirmation-20261005-r1.service` started at
2026-10-04 16:51:41 UTC with a 189,063-second supervisor limit and 10-second
cleanup allowance. The first worker verified memory growth and emitted GPU/XLA
evidence. These are startup checks; no complete-slot or release claim is made.

The numerical source, criteria and defaults are unchanged. Preserve every
planned outcome and all verified members. Follow
`artifacts/hmc-v7-release-2026-10-02/confirmation-run-01/progress.json`, then
audit the terminal evidence and costs before P5. The service reservation
includes all its child work; charge its enclosing duration once. The passed
preflight and exact command are under `confirmation-design-01`.

Unattended supervision is specified by the
[October 5 monitoring plan](artifacts/hmc-v7-release-2026-10-02/confirmation-monitor-plan-2026-10-05.md).
Its persistent observer writes `monitoring-01/status.json` and `status.md`,
checks health every minute and requests desktop error/completion alerts.
Only untouched pre-import resource deferrals can receive one secondary recovery
attempt after the primary service ends, within its remaining reservation.
Secondary results never replace primary outcomes or alter the release rule.

## Prelaunch October 5 checkpoint

Status: release **pending**. Both complete source-23 development price vectors
and independent audits pass. The resource-observed repeat completed in
3,657.258328 GPU seconds, retained all 22/19/17 verified members, and exactly
matched all 31,072 original raw trials, candidate inventories and states.

The repeated 96-slot point forecast is 32.51 GPU hours; the predeclared 20%
reserve, readiness and terminal allowances bring it to 40.78 hours. The
October 5 grant adds 24 GPU hours; 55.10 authorized hours now remain, with
14.32 hours beyond that scenario. It fits inside the release allocation alone.
No additional CPU grant is assumed. Confirmation remains unfrozen and
unlaunched. Both whole vectors and margin sensitivities are retained in
`artifacts/hmc-v7-release-2026-10-02/resource-price-affordability-grant-01`;
the current amendment is `artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-05.json`.

The bounded host-affinity diagnostic completed. All 36 native calls and
1,152 full trial records matched exactly. The quieter cores made warm record
assembly slower in both repetitions for all three families, and native timings
overlapped. No affinity or numerical-runtime change is adopted. The initial
resource deferral and successful retry are charged; all reservations are
released and no GPU run remains active.

P4 now requires the final 96-slot design: freeze independent seeds disjoint
from development, unchanged configurations, tested supervisor, exact budget
and fresh output root; recheck source, GPU and storage readiness, then execute
and audit all slots. The measured funding gap is cleared. Future costs remain
uncertain: the earlier slower vector with the same allowances requires 60.49
hours and exceeds the balance. Preserve the hard budget and classify failures,
timeouts and deferrals on the original denominator. No speculative optimization
or further complete pricing is required before design preparation. P5 remains
the final source/evidence, regression/documentation and local support decision.

The [master](bayesfilter-hmc-repair-master-program-2026-09-16.md) and
[current continuation record](artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-05.md)
contain the current evidence contract, skeptical audit and exact budgets. The
[October 4 record](artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-04.md)
preserves the completed experiments and decision/inference-status tables.
Source-23 remains numerical authority. Defaults
and reporting-only tuning R-hat are unchanged; every verified member remains.
Earlier checkpoints below remain historical evidence.

## Release question and scope

Can an explicitly selected v7 policy reliably deliver a checked set of fixed
HMC kernels, with defensible finite-trial acceptance uncertainty, complete
candidate-specific repair, independent verification, reproducible recovery and
an honest handoff to posterior assessment?

The intended first release supports the existing two public tuners on exact
TensorFlow/TFP value/score targets, four frozen starts, float64, XLA, prepared
ordinary coordinates or a supported frozen transform. The statistical target
is the conditional fixed-horizon mean, not stationary acceptance. R-hat and
temporal diagnostics remain reporting-only during tuning. Every verified
member survives; there is no diagnostic ranking or automatic nominee.

Automatic windowed preparation must have an actual v7 integration test, but
automatic-preparation reliability and changing numerical defaults are separate
promotion questions. Supplied analytic funnel maps do not establish learned
NeuTra training quality. A nonlinear sigma-point likelihood is a finite
approximation whose derivative can be checked, not an exact posterior oracle.
Unwhitened centered funnels, native hard-support rejection without bound
telemetry, missing/diffuse/singular SSM variants and the unavailable K6 joint
oracle remain explicit exclusions. They do not become universal release gates.

## Evidence contract and finite exit criteria

The comparator is the October 2 v7 implementation and its frozen evidence,
not a legacy v5/v6 run. Existing 441 distinct passing checks, 28,672 synthetic
searches, four recovery cases, four trusted parity cases and three native GPU
deliveries remain reusable for their stated scope. They do not close the new
requirements below. No method or kernel superiority claim is sought.

| Existing phase / release gate | Required new evidence | Pass condition |
| --- | --- | --- |
| P0/P1 identity and release assembly | Freeze intended sources and dependencies; preserve v5/v6/v7 payload readers; test a fresh source tree using documented environment/import workflow | Required modules/tests/data present, no accidental dependency on dirty unrelated work; status metadata cannot alter policy hashes; regression command reproducible |
| P2 full procedure | Actual v7 preparation, primary L=(3,5,9,13,18,25), poor epsilon proposals, immutable same-L repairs, declared refinement, fresh verification and export/reload on both public routes | Required stages execute; initial peers are accounted for; multiple survivors are retained where the frozen positive fixture requires them; interrupted repair/verification equals uninterrupted numerical evidence |
| P2 numerical boundaries | Independent high-precision bounds near qualification/preferred thresholds, endpoints and evidence limits, on CPU/XLA and trusted GPU/XLA | Returned endpoints contain independent reference bounds within conservative arithmetic direction; ambiguity never gains membership; tested envelope is recorded rather than claiming a hardware theorem |
| P4 reliability and cost | Three frozen prepared families: QR LGSSM, nonlinear SSM and residual-whitened funnel; initial proposal 32 independent full-search tuning seeds per family | Complete planned denominator; one-sided simultaneous 95% lower delivery bounds across three families exceed inherited 0.80 easy-case requirement; a successful slot completes within its declared process cap and all attempts fit the total campaign budget; failures remain counted |
| P4 posterior interface | Reloaded members feed sequential warmup/retained assessment; repeated analytic location/Kalman checks plus existing missed-mode negative control | Warmup excluded; accepted tuning cannot bypass posterior veto; reference discrepancies reported with MCSE; no SBC or coverage claim without a matched simulator and adequate independent replication |
| P5 terminal release decision | Release manifest, scoped regressions, official chapter/agent reference agreement and remaining-limit table | All required scoped gates pass or release stays pending with a precise failed gate; terminal result separates engineering, numerical and statistical evidence |

The 32-run allocation is an engineering proposal, not a new numerical default.
If every run succeeds, the simultaneous lower bound is
`(0.05/3)**(1/32) = 0.879898897...`; this follows directly by inverting
`P_p(all 32 successes)=p**32` with Bonferroni allocation. With failures, use
the exact binomial lower bound on the original denominator. A resource-deferred
or inconclusive slot is not successful delivery. Report numerical/algorithmic
and resource causes separately. Development seeds cannot enter confirmation.

The binomial statement concerns the probability, over independent seeds, that
the frozen numerical search delivers a member when its declared work is fully
executed. It is not a common
deadline-success probability under arbitrary shared-GPU load. If a resource
interruption hides a slot's numerical outcome, count that slot as unsuccessful:
the observed success count is at most the complete-work success count, and the
binomial lower bound is monotone in that count. This supplies a conservative
bound for numerical delivery without assuming noninformative resource censoring.
All slots and enclosing costs still enter the operational budget assessment;
neither that bound nor a single idle-device price proves a future runtime guarantee.

First price the **actual full search** with the intended candidate cap and
refinement/repair workload. Existing M2/L1 prices cannot fund M100/full-grid
claims. Freeze the final configuration, full independent seed inventory and
total price before confirmation. If unaffordable, document that result before
revising the design: do not silently shrink the denominator, use only easy
seeds, omit L values or call single-pair replication full-procedure evidence.
Any narrower release proposal must state exactly which reliability claim it
can support. Price near-unit, small-noise, long/multivariate variants separately;
they cannot inherit a cheap-model runtime guarantee.

## Budget, launch and result records

The October 2 CPU ledger leaves 70,536.183646 worker seconds after conservative
earlier debits. The reconciled GPU balance leaves 67,612.316725 uncommitted
seconds after the running SSM reservation, protected allowance and earlier
600/1,800-second acceptance charges. Do not debit prior nested work again.
The old-policy SSM service has since closed and its accounting is reconciled;
it cannot validate v7. Its unused reservation adds 1,665.699 seconds to the
7,612.317 initially unallocated GPU seconds, now 9,278.015 outside this release
allocation. Its original ledger and every failed attempt remain preserved.

The initial release reservation was **21,600 CPU worker seconds** for engineering,
numerical references and CPU integration, with at most four CPUs per worker.
On October 5, **3,600 previously authorized CPU seconds** were transferred for
monitoring and terminal audit, bringing this allocation to **25,200 seconds**.
The outside CPU balance is now 45,336.183646 seconds; no new CPU grant was added.
The initial GPU ceiling was **60,000 seconds** for full-search pricing, confirmation
and terminal checks. The October 4 and October 5 user grants each add
**86,400 seconds**, making the active release ceiling **232,800 GPU seconds**.
They are distinct grants, each recorded once in `allocation.json`. The historical initial
unallocated balance was 7,612.316725 seconds; closed-service reconciliation
raised the untransferred balance to 9,278.015433 seconds. These are
convenience ceilings within existing authorization, not measured prices.
Maintain a separate release ledger rather than editing the running service's
in-memory GPU ledger. Charge failed and deferred attempts as well as successes.
Each launch uses a unique directory under
`artifacts/hmc-v7-release-2026-10-02/`; store source hashes, commit, environment,
command, seeds/data/config, hardware settings, plan, result and enclosing time.

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, bounded TF/BLAS threads and
CPU affinity. CPU reference commands set `CUDA_VISIBLE_DEVICES=-1` before
import. GPU commands use trusted execution, incremental allocation configured
before import/initialization, actual per-chunk device evidence and XLA. Respect
the existing shared-GPU policy, preserve progress under contention and do not
confuse a busy-device preflight with a numerical failure. Never resume a
source-bound checkpoint after changing its source closure.

A failed candidate is a promotion veto and a repair trigger, not a continuation
veto. Stop a run for exhausted budget, invalid target/evidence, unaccounted work,
source mismatch or violated memory policy. Localized harness/resource repairs
may retry under unchanged science and budget with fresh attempt records.

## Skeptical pre-execution review

The audit rejected these material shortcuts before implementation:

* The earlier model checks used mostly one nominated L=1 pair and no repairs.
  They cannot establish the full public search. The new P2 gate requires it.
* Nominal M100 allocation and synthetic calibration do not price real M100
  numerical execution. Freeze and price the actual release workload first.
* Prepared geometry is not automatic preparation; the two coverage claims are
  separate. Poor geometry may correctly yield no valid member.
* The exact probability inequality does not prove float64 inversion accuracy.
  Independent boundary references and explicit tested limits are required.
* `default_promotion_status` participates in the v7 policy hash. Changing that
  string to mark a release would break saved identities. Preserve the codec;
  expose support status outside numerical identity or use a future version.
* Core v7 modules and tests are currently untracked. An import from this dirty
  worktree is not a clean-source reproducibility test. Assemble explicit files
  and data, excluding unrelated q20/NeuTra work; no invented packaging tooling.
* Count every confirmation slot and preserve failed criteria. Test counts,
  reference smokes and acceptance delivery cannot establish posterior coverage.
* Available resources exclude the running SSM reservation. The earlier
  727.615-second GPU total only prices three single-pair runs; multiplying it by
  32 is not a full-grid forecast.

The revised staged plan passes this audit for **development and pricing**.
Confirmation remains conditional on its frozen design passing affordability
and source/device preflight. This condition is an engineering/scientific check,
not an additional human approval ceremony.

| Material choice | Provenance / status | Failure risk and earliest check |
| --- | --- | --- |
| Six-value primary L grid and M100 | Existing ordinary convenience settings; baseline | Excessive work or resonance; one measured full-search price and per-L outcomes |
| Qualification [.55,.85], preferred [.65,.75], separate alpha=.05 allocations | Existing explicit v7 settings; hypotheses | Excluding useful kernels or mistaking two statements for joint .05; independent model outcomes and unchanged calibration |
| W=3, T=65 initial reference ladder | Existing v7 diagnostic fixtures; convenience baseline | Different finite-horizon estimand and excessive work; preserve exact identity, report temporal diagnostics, no stationary claim |
| Repetitions 32--256 for development, then frozen confirmation schedule | Existing positive tests extended for larger candidate allocation; hypothesis | More abstention under M100; price full search and retain inconclusive results |
| Up to 16,384 repetitions, M<=100, fixed four-bet grid for boundary audit | Bounded proposed numerical support envelope | Undetected arithmetic failure or unsupported extrapolation; high-precision adversarial cases at limits, report scope |
| 32 independent seeds/family, 0.80 lower delivery requirement | Engineering allocation and inherited easy-case requirement | Unaffordable or underpowered with failures; exact lower bound on all planned slots |
| Four-MCSE analytic posterior check | Existing diagnostic tolerance, not finite-sample coverage theorem | MCSE error or multiple testing; repeated errors/MCSE and separate convergence/reference assessment, no calibration claim |

## Execution order and refresh rule

1. Extend maintained model execution to serialize full-search settings; add
   actual preparation/repair/refinement/recovery regressions and high-precision
   boundary references. Reuse completed evidence where its scope matches.
2. Freeze source and run the focused suite in a fresh tree. Record exact
   dependency closure and legacy codec tests. Diagnose failures before pricing.
3. Run bounded full-search prices and an affordability calculation. Freeze a
   feasible confirmation design before spending its reservation; otherwise
   record the exact funding/algorithmic gap and pursue independent gates.
4. Execute confirmation and posterior-interface checks, retaining all outcomes.
5. Reconcile costs and terminal evidence, update the official guide and registry
   only to the supported status, and write the release decision.

After each step update this plan's execution record, the master's leading
status and the release `progress.json`. A terminal release record needs a
decision table, inference-status table and strongest alternative explanation.
Do not end at a successful launcher or substitute more planning for the work.

### Pricing-triggered implementation diagnosis

The first M100/six-L Gaussian development run observed 418 native chunks in
237.641 enclosing seconds, but only 7.499 seconds inside native calls plus
their immediate serialization. This is descriptive evidence of host overhead,
not a GPU speed comparison. It triggers a bounded checkpoint profile before
confirmation: load a consistent durable checkpoint, profile repeated writes
to a fresh diagnostic directory, and identify time spent hashing, formatting,
revalidating and writing. Budget: at most 180 CPU worker seconds within the
18,000-second reservation. No numerical criteria, evidence or seed changes.

Skeptical audit: a live checkpoint is atomically replaced and references
immutable existing evidence, but the diagnostic must load and validate that
snapshot before profiling. Do not edit source-bound runtime modules while the
original development run is active. A performance repair must retain full
validation on explicit save/export/reload, all attempted-work charges, raw
trials and interruption recovery. A weaker evidence check is not a speed fix.
Record a new source/config identity for any subsequent attempt and compare
identical native streams and decisions where identities permit it.

The profile completed in 24.820 CPU seconds. Three full checkpoint writes
spent 3.342 profiled seconds; JSON serialization consumed 2.262 seconds and
rehashing accumulated immutable evidence 1.639 seconds (nested, not additive).
At this snapshot the checkpoint's observations repeated about 2.6 MB of trial
health, and immutable numerical evidence exceeded 100 MB. Repeating those
checks before and after each short native chunk makes cost grow with history.

Reviewed repair: keep the existing checkpoint schema and contents; write compact
JSON, use the equivalent strict JSON-native checksum for normalized checkpoint
bodies, and let internal within-work chunk saves reuse already persisted
unchanged evidence. New/changed files must still be validated. An explicit
checkpoint write remains a full live-integrity check by default. The internal
fast-save scope must exit before a completed-work observation, pause/failure
save, final result, export or reload; those boundaries recheck all evidence.
This changes validation cadence within a work item, not admission, seeds,
durability, the statistical method or artifact authority.

Pre-implementation audit: a naive hash cache could hide live mutations. Test
that full saves still reject mutations after a fast save, that a changed or
missing evidence file is repaired or rejected, that failures leave fast mode,
and that charge-before-call remains durable. Compare decoded payloads and
hashes against the old serializer; repeat actual process recovery/accounting
regressions. No journal, new authority token or new public schema is needed.
The old running price keeps its source until it stops at its original cap.

The original price stopped at its 900-second numerical ceiling; enclosing time
was 931.272 CPU seconds including validation/closeout. No verified member was
delivered before the cap. This is an incomplete search/cost failure, not
evidence against HMC or a reason to weaken admission. The checkpoint repair
was applied only after that process ended.

Fresh-source validation exposed a second concrete packaging gap: 143 checks
passed, but 16 SSM import/validation cases failed because the native
`_symmetric_sylvester_ops.so` was absent. The source assembler now includes the
tracked CMake inputs and an explicit build in the new tree using the selected
TensorFlow environment. It records the build command and library checksum;
it does not copy an unrecorded binary from the dirty worktree or install packages.
Build ceilings of 180 seconds for configuration and 420 for compilation are
debugging caps within the CPU reservation. The prior failed tree is preserved.

The new native build passed. Both actual repair/verification interruption
tests now resume in fresh processes with identical trial samples, scores,
decisions and two retained members; the interrupted attempt remains charged.
One serializer test assertion needed correction because the returned terminal
result and the last callback had different elapsed-time fields. It now compares
full and incremental serialization of the same explicit state.

The second six-L price runs on frozen `source-02`. Its progress still identifies
a redundant payload problem: every controller observation embeds the full
health diagnostic for every trial, although every detailed record already
exists under immutable numerical evidence. The next bounded repair keeps those
raw records and decision rules intact but summarizes health counts in new
`hmc_replicated_acceptance_evidence.v2` observations. The numerical policy hash
and checkpoint schemas stay unchanged. Recompute old v1 analysis in its old
form on historical readback; test old/new decision equality and full recovery.
This is internal evidence compression, not removal of any raw scientific data.
The consumer trace found no runtime reader requiring the duplicated health list.
The frozen running price is unaffected by edits to the active tree.

The retained-member trace also found that each validation call recomputed every
numerical analysis in its inventory scan and then repeated the same computation
in its receipt scan. Reuse the just-checked analysis within that one call only;
do not cache it across exports, reloads or live mutations. This preserves every
receipt/seed/hash/endpoint and shared-invalidity check. A call-count regression
must accompany the existing mutation and retained-replay tests. The historical
health test compares JSON-normalized records, because decoded lists and freshly
computed tuples are equivalent in the existing wire format.

### Concrete GPU pricing inventory

After the compact-observation, native-build and focused regressions pass, use
`scripts/run_hmc_v7_release_prices.py --source <frozen source> --output <new root>
--gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c --wall-seconds 1800
--budget-seconds 6000`. The three predetermined development cases are QR LGSSM,
nonlinear SSM and residual-whitened funnel, with seeds `(20261002,2501..2503)`.
They use the complete six-L/M100 configuration in
`bayesfilter.testing.acceptance_release_validation`, including repair and
refinement, not the old single-pair controls. The source/runner/config are
frozen, memory growth is verified, every persisted native chunk must identify
GPU/XLA execution, and all outcomes and closeout costs remain recorded.

The 1,800-second per-search and 180-second closeout caps are development cost
ceilings, not demonstrated adequate runtime. The 4,096 MiB free-memory screen
is an inherited conservative launch screen, not proof of capacity. Shared GPU
utilization is recorded and is not a reason to reinterpret a timeout as a
numerical failure. Resource deferral leaves the price unavailable. The queue
stops on unknown harness failures; ordinary no-delivery/resource outcomes keep
their slots and do not invalidate independent families. Total GPU pricing is
capped at 6,000 seconds from, not in addition to, the 60,000-second reservation.
No confirmation starts automatically from this exploratory price.

### Full-search scheduling defect and reviewed P2 repair

The two complete CPU pricing artifacts contained screened measurement survivors
but no fresh verification before the time cap. Source tracing confirms that
`_next_cohort` orders all measurement work, including optional extensions and
new repair descendants, ahead of every verification. Reserved verification
work therefore does not prevent practical starvation under a wall budget.

Repair new v7 searches with controller policy version 4: finish the active
cohort and every ready non-repair initial measurement/pilot first, then give
screened members their pending fresh-verification ladder before optional
measurement extensions and repair-child measurements. Keep all candidate
records and continue the rest of the search; this is not first-pass stopping
or a ranking. Legacy v5/v6 searches and resumed v3 controllers retain v3 order.
No new public entry point or scientific threshold is introduced. Old sources
and active v3 prices stay frozen; do not resume them under changed source.

Skeptical review: arbitrary adaptive ordering is already covered by the frozen
candidate/rung error allocation and independent verification streams, but
must not skip a primary peer, award membership before verification, add looks,
reset budget, or silently reorder a historical checkpoint. Require deterministic
tests for repair/extension starvation, mandatory broad coverage, inconclusive
verification extension, interrupted v3/v4 reconstruction, unknown-version
rejection and equal terminal pair sets under sufficient work. Repeat actual
v7 repair/verification recovery and affected legacy/controller/calibration tests.
Use a fresh frozen source for later confirmation. Earlier prices retain their
cost and incomplete-outcome meaning and do not become v4 confirmation evidence.

### P2 health audit: chance endpoint return is not immobility

The source-04 regression has 187 passes and one failed original K0 posterior
handoff case (replication 2). Its acceptance intervals qualified, but trial 68
of verification had normalized first-to-last displacement 0.0000578732 in one
chain, below the inherited 0.0001 screen. All four adjacent movement rates were
0.640625--0.734375; short-cycle return fractions were 0.047619--0.126984, far
below their 0.95 veto. The evidence is a moving, nonperiodic path with two close
endpoints. It does not support calling that path immobile. R-hat played no part.

K0's geometry is derived from its exact quadratic native score. Shifting the
data changes its location while whitening leaves the same standard Gaussian
geometry. Attributing this failure to poor geometry was wrong. Trying smaller
panel shifts and new seeds gave 25 passing focused checks on source-05, but
that selection cannot repair the failure; the original panels and seeds have
been restored. Preserve both attempts as development evidence and their costs.

For an independent standard-Gaussian endpoint comparator, the endpoint
difference has variance 2 and the probability that its absolute value is below
c is erf(c/2). Thus even with known scale the inherited c=0.0001 has positive
false-rejection probability. Requiring no such endpoint return among N
independent paths gives rejection probability 1-(1-erf(c/2))**N, approximately
0.109 at N=4*256*2. This is an exact independent-endpoint comparator, not a
calibration of dependent HMC paths with sample-estimated scale. It explains why
unioning a chance endpoint coincidence across trials is not a movement proof.

Reviewed repair: keep the library's legacy policy/default and payload reader
unchanged. Explicit release-validation configurations set the already supported,
identity-bound `min_normalized_return_displacement=0.0`; zero has the deliberate
meaning that endpoint distance is reporting-only. Adjacent movement, repeated
states, repeated short-cycle recurrence, native divergence, target validity,
discarded-prefix health and all acceptance criteria remain active. This is a
declared experimental health-policy change requiring fresh numerical evidence,
not a retroactive pass for any old candidate. The supported release scope must
name this explicit configuration. No source-03/04/05 result qualifies that new
configuration, and no default change is proposed.

Evidence contract: compare the saved failed native trial and deterministic
moving-return paths against the legacy endpoint veto. Positive criterion:
the explicit profile reports the small endpoint distance without a movement
veto. Promotion veto controls: frozen paths, one frozen coordinate, persistent
short cycles, divergence and nonfinite target/state still fail. Check legacy
payload identity/behavior, then rerun the original three K0 panels/seeds and
their posterior reference checks. No posterior coverage, stationary-acceptance,
or universal false-veto claim follows. Preserve the failed trace fixture,
test JUnit/log/receipt and new frozen source. Bound this diagnosis/repair at
1,800 CPU worker seconds within the release reservation. Do not select new
seeds or tune panels to pass. If any original panel still fails, inspect its
actual veto or uncertainty and preserve it in the delivery denominator.

Skeptical review: disabling a coincidental endpoint veto could hide a true
cycle. The repeated-lag control must therefore remain a veto, including a path
whose endpoint does not return; a single distant endpoint must not certify
mixing either. Only the existing explicit configuration changes, so old frozen
sources and codecs remain interpretable. The newly configured policy has a
different identity and cannot resume an old checkpoint. This amendment passes
for bounded development; it does not pre-certify release or affordability.

### P4 audit: distinguish member delivery from a complete search price

The generic model diagnostic deliberately defines `positive_delivery` as at
least one checked export, even when budget stops the remaining search. The
pricing runner reused that predicate to label its search `complete`. That
label is wrong for this release's full-procedure requirement. No confirmation
has used it, and the completed source-03 attempt delivered no member anyway.

Repair only the pricing interpretation: a usable full-search price requires
`completion_status == "complete"`, successful checkpoint reconstruction,
positive model delivery, and equality of the exported and verified member
sets. Preserve partial exports as useful tuning results, but classify their
price as incomplete. Keep the live frozen runner unchanged and independently
audit its eventual result with this rule. Test a partial search with verified
exports, a complete empty search, omitted member exports, absent reconstruction
and a complete all-member export. A mislabeled partial search must not trigger
a cost forecast or confirmation. This is a reporting/campaign-decision repair,
not a numerical or public-tuner change. Budget at most 120 CPU worker seconds
within the existing reservation for the focused regression.

Skeptical audit: partial delivery is valid evidence for the exported kernels,
so the repair must not revoke membership. Conversely, a complete search can
deliver nothing; completion alone does not satisfy the positive delivery gate.
Use controller completion and checked exports as separate predicates. The
corrected rule preserves failed slots and original costs and passes this audit.

### Native-cost diagnostic before another GPU price

The source-05 QR checkpoint has substantial time inside the native graph
(for example, 614 chunks accumulated 1,304 seconds inside native calls plus
immediate serialization). Trusted process accounting also shows two other
applications sharing its GPU. The existing GPU queue keeps its frozen settings
and cap. Do not attribute all remaining cost to checkpoint writes or launch
another identical long price without a discriminating diagnosis.

Run one source-06 QR full search as an explicitly CPU/XLA reference, using the
original development seed `(20261002,2501)`, full six-L/M100 profile and repaired
endpoint policy. Bound numerical work at 900 seconds plus 180 seconds for
reconstruction/export, charged within the CPU reservation. Preserve every
outcome and per-chunk native/diagnostic/persistence timing. This tests whether
the host path remains a practical bottleneck and supplies procedure evidence;
it does not price GPU operation, qualify the GPU default or estimate multi-seed
delivery. A timeout is still incomplete. No new panel, seed, shorter trial,
smaller grid or changed acceptance interval is permitted.

Skeptical audit: CPU and GPU differ in hardware and concurrent workloads, and
source-05 and source-06 differ in health policy. Their timings cannot support a
method superiority claim or a causal device-speed ratio. Native timing buckets
and a complete CPU procedure can nonetheless distinguish unresolved host cost
from a predominantly GPU-execution problem. Future optimization requires its
own same-target numerical checks. This bounded diagnostic passes the audit.

### P2/P4 closeout throughput repair trigger

The source-06 CPU reference reached refinement and verified six of 26
candidates before its 900-second numerical ceiling. At the 1,080-second process
ceiling it had written only the first compact member and a 725 MB shared evidence
bundle; no terminal result exists. Source-05 GPU pricing also remained incomplete
in all three original slots (5,857.215 seconds total). Preserve these outcomes;
do not start 96 confirmation searches from their partial prices.

The next discriminating check profiles one fresh reload of that saved CPU member,
using its original source-06 target and environment. Preserve the result,
full cProfile data and separate JSON/hash/health/reconstruction time. A 300-second
internal diagnostic cap and 330-second enclosing cap bound the check inside the
CPU reservation. If interrupted, its profile describes only work reached before
the cap and supplies no successful-reload evidence. A completed reload must
retain all existing stream, cost, raw-trial, endpoint and shared-invalidity
checks. No changes to the running scientific evidence or acceptance thresholds
are allowed.

Skeptical audit: profiling overhead and host load prevent interpreting these
times as an uninstrumented production price. The purpose is to identify repeated
work before choosing a repair. Prefer call-scoped reuse of already checked
immutable evidence over changing scientific gates or introducing a new archive
format without need. A repair must preserve old readback, detect live mutations,
validate every retained member and pass actual process recovery. Native GPU
execution remains a separate cost; repairing reload does not alone establish
affordability. This diagnostic passes the audit; implementation follows its
measured attribution, not a guessed bottleneck.

The reload passed in 194.595 instrumented seconds (199.174 enclosing seconds).
It made 131,690 generic hash calls, spending 67.408 cumulative seconds in
normalization/hashing, and decoded 104,579 tensors. The 725 MB bundle was parsed
only once for this member, but the harness repeats the entire reload independently
for every survivor. The profile is descriptive, not an uninstrumented price.

Reviewed implementation, initially under a 1,600 CPU-worker-second focused-check allocation:

1. Use the already tested strict JSON hash for JSON-normalized numerical records
   and decoded artifacts. Keep generic identity hashing for arbitrary API inputs.
   Valid JSON identities must be byte-for-byte unchanged; nonfinite or unsupported
   objects must fail, never receive a fallback representation.
2. Reuse tensors decoded by chunk validation in the immediately following trial
   reconstruction. The reuse is confined to one validation call; every later
   validation still rechecks raw bytes, hashes, identities, counts and continuity.
3. Provide grouped retained export/reload helpers using the existing shared
   evidence-bundle/member formats. Validate a group's common numerical evidence
   once, then check every requested member's verified status, endpoint and hash.
   The individual-member API keeps its behavior. The model harness uses grouped
   export/reload for all verified IDs; it cannot omit a survivor or skip checks.

Pre-implementation skeptical review: a persistent trust cache would hide later
mutations and is rejected. Group reuse ends when its single synchronous call
returns; subsequent replay/run/export validates again. A shared-invalid result
must reject the whole group, including formerly verified members. Mixed targets,
missing bundles, altered endpoints/IDs and corrupted raw traces remain failures.
Check old single-member compatibility, equality of grouped and individual
endpoints/member identities, one shared validation per group, revalidation after
live mutation, and real interrupted-trial recovery. No new numerical thresholds,
seed selection, source-bound resume, or evidence-file format is introduced.
This repair addresses measured host work; independent GPU delivery and affordable
confirmation remain open and require new pricing on a fresh source.

The first repaired-source suite passed 78 checks in 580.148 enclosing seconds.
Including focused corrections and source builds, allow a 2,000-second total
implementation-check allocation inside the unchanged 18,000-second CPU
reservation. This revised convenience cap funds the remaining legacy/checkpoint
suite and four real QR/nonlinear process-recovery checks, rather than omitting
them to fit the initial estimate. The final source also applies the same
JSON-native hash to decoded checkpoint evidence; old valid JSON identities stay
unchanged. Rebuild and verify that source before pricing.

### Native graph comparison while the repaired price runs

Source-08's live QR price records substantial chain-call time: for example,
128 L=13 chunks accumulate 110.689 seconds in the chain call and 2.877 in
post-chain diagnostic capture. This descriptive breakdown does not establish
the cause. The release harness selects dynamic leapfrog-count graph reuse;
static per-L runners are an existing supported alternative whose GPU cost has
not been measured on this target.

Run one diagnostic on frozen source-08 comparing those two existing runner
options at L=3 and L=25, with the same QR target, data, affine preparation,
epsilon=.95, four starts and 68 transitions as the live configuration. Use one
initial call and three alternating paired warm calls per L. The static runner
compiles separately at each L; the dynamic runner reuses its first graph at
the second L. Record that distinction rather than charging it a second cold
compile. Seeds
`(20261003,2600..2607)` are new development seeds, never confirmation. The
small repeat count and shared GPU imply descriptive timing only; the purpose
is to locate a possible cost repair, not establish runtime superiority.
Record complete compile/call/diagnostic/enclosing times, device, verified
memory growth, graph counts, source hash and peak allocator use. Require
identical shapes, finite traces and numerical parity against the existing
dynamic path using the existing runner-reuse regression tolerance. A mismatch
vetoes adopting the option and triggers diagnosis. Any later full-search
configuration change needs a fresh source/configuration and new price.

Budget at most 240 GPU seconds for one attempt within the existing 60,000-second
reservation. This convenience diagnostic ceiling includes compile and closeout;
timeout produces no timing conclusion. Do not touch the running three-model
price or infer independence from shared-resource timing. The pre-execution audit
passes: the comparator changes only an existing execution option, tests both
ends of the declared L grid, retains the exact numerical problem, includes
cold cost, and cannot promote a configuration without complete-search evidence.

### Repaired full-procedure CPU reference

Run the existing `cpu_full_search_price.py` against source-08 once, using the
same QR panel, `(20261002,2501)` seed, six-L/M100 search and 900-second numerical
plus 180-second closeout ceilings as source-06. Charge at most 1,080 CPU worker
seconds within the remaining release allocation. This answers whether the
measured host repairs now permit full numerical execution and all-member
handoff on the reference device. It cannot establish GPU affordability or
multi-seed reliability. The original source-06 failure remains counted.

Skeptical audit: compare complete procedure and raw outcome inventories, not a
single successful export or process exit code; use the same complete-search
predicate as GPU pricing. Device load and changed host implementation prevent
an uncontrolled timing difference from proving speed superiority. This is a
bounded reference check justified by the host-code changes since the earlier
failed reference; it leaves the live GPU source/configuration and confirmation
denominator unchanged.

### Repaired GPU closeout diagnosis

Source-08's QR price exhausted its 1,980-second process ceiling after writing
both verified member files and their 326 MB shared bundle. Its recorded native
chunks contain 1,450.470 chain-call seconds and 36.507 diagnostic-capture seconds.
The complete search and closeout remain unpriced. Profile the grouped reload
of those two saved members in one fresh process using the exact original
GPU/source/configuration. This performs no new HMC sampling. Bound it at
360 GPU seconds, including a 330-second profiler deadline; save partial profile
data on interruption. Charge it inside the release ceiling and retain failure.

The exact common-bundle load is the comparator; successful reconstruction of
both original verified identities is the engineering pass criterion. Timing
buckets identify a repair trigger, never a speed or release promotion. A byte,
seed, source, member, endpoint or device mismatch is a veto. The skeptical audit
passes because it interrogates the actual timed-out closeout and keeps all
scientific data/decisions fixed. If it passes, a profile alone still does not
make the unfinished full search complete. Concurrency and profiling overhead
remain explanatory limitations.

### Measured health-reconstruction repair

The saved GPU group did not finish reconstruction within its 330-second
profiling cap (334.922 enclosing seconds). The partial profile records 136,513
tensor-to-host conversions consuming 122.981 cumulative seconds, including
50,839 scalar float conversions (58.073 seconds) and 40,066 Boolean conversions
(43.629 seconds). Strict JSON hashing accounts for 8.078 seconds. These buckets
overlap; they are not additive, and the partial profile does not price a complete
reload. The source-08 CPU reference also reaches refinement/six verified members
but again hits the 1,080-second process cap. All outcomes remain incomplete.

Repair this measured repeated work without changing the health rule: compile
the repeated finite-value/Metropolis consistency reductions into a stable
signature TensorFlow graph, then transfer its Boolean vector once. Materialize
each already-computed diagnostic vector to Python values once at the reporting
boundary, preserving its individual values and serialized fields. Keep model
status schema checks, prefix health, nonfinite-state/score/momentum checks,
divergence, movement and cycle vetoes. Preserve the numerical policy and wire
schemas; only a fresh source can run the repair. No NumPy computation, pfor,
new acceptance bound, seed substitution or persistent trust cache is allowed.

Evidence contract: compare the helper against the frozen source-08 health
implementation on healthy and corrupted traces, including first-step rejection,
nonfinite initial state, proposal displacement overflow, momentum and score
failure, malformed/failed target status and native divergence. Require exactly
equal reason codes and serialized health outputs. Check CPU and trusted GPU
execution and bounded tracing, then repeat affected public candidate/replay,
endpoint and recovery tests. Profile a fresh identical-sized fixture or a
same-source diagnostic copy of the saved raw traces; do not resume the original
checkpoint with changed source or call a source-edited copy an original
admissible artifact. A parity failure blocks adoption and triggers diagnosis.

Reserve at most 1,800 CPU worker seconds and 480 GPU seconds for implementation,
parity and one bounded cost diagnostic inside the unchanged release ceiling.
These are convenience investigation caps. Before another full price, record
whether the repaired profile materially removes the observed bottleneck and
whether native execution still makes confirmation unaffordable. The skeptical
audit passes: the change targets measured transfer/dispatch overhead, preserves
the actual arithmetic and all vetoes, and tests the complete caller path rather
than merely a new unused helper. It cannot by itself close full-search or
independent-replication gates.

### Remaining native cost: bounded batch-capacity diagnostic

The source-08 QR price records 1,450.470 chain-call seconds for 1,492
four-chain chunks. The same-sized CPU reference records 56.021 chain-call
seconds for 3,259 chunks, although source/device/streams and concurrent load
differ. These observations cannot support a device-speed claim. They do show
that fixing host reconstruction alone cannot establish the proposed 96-search
confirmation's affordability. Repeating all three long prices now would leave
the native-work explanation unresolved.

Before changing the numerical runner, measure the existing frozen source-09
runner on the original QR and nonlinear targets at L=25, with the original
epsilon, four starts, W=3 and T=65. Compare calls with one, eight and 32 copies
of the four-start bank, using the runner's existing independent chain-row
semantics. These batch sizes are convenience capacity probes, not proposed
statistical defaults. Use one cold and two warm calls for each shape, seeds
`(20261003,2800..2817)`, and record all calls. The six shape/model cells contain
no candidate search, no acceptance admission and no reusable tuning artifact.
Keep dynamic-L mode as in the current price. Record compile/execute and
diagnostic times, total enclosing time, peak memory, live GPU load and verified
memory growth. Divide time by copies only to describe cost per four-start bank;
the fixed number of calls and small sample cannot establish a speed ranking.

The engineering question is whether larger independent-chain batches remove
the observed tiny-call bottleneck enough to justify a separately audited trial
batching implementation. At the unchanged initial states, compare batched
value/score and target-status rows against single-bank rows exactly. Require
the declared shapes, finite native trace and correct Metropolis state updates.
The current scalar seed API does not reproduce individual-trial streams when
the state shape changes: do not assert trajectory parity or reuse its outputs
as original-seed evidence. Statistical independence is an explicit property
that a later implementation must justify from the product transition, its
random-number layout and recorded trial identities, not from this timing probe.

Reserve at most 600 GPU seconds within the unchanged release ceiling. One
failed setup repair may reuse the remainder in a new directory; no timer reset
or unrecorded retry. Timeout or mismatched rows/invalid transitions vetoes
adoption and triggers diagnosis; it does not reject HMC. If capacity remains
poor, preserve that result and report the measured affordability gap. If it
is promising, first specify bounded batching, per-trial health and scores,
fresh search/verification streams, durable charge-before-call, loss/retry
accounting, exact raw-evidence reconstruction and legacy single-trial behavior.
Only after that review and focused numerical/recovery tests may a fresh full
search price use batching. Do not shrink L coverage, horizons, candidate caps,
panels, seed inventory or statistical criteria to close the cost gate.

Skeptical audit: this is a native-capacity probe, not the missing complete-search
price or confirmation. It tests the expensive L boundary on both state-space
targets, includes compilation and diagnosis cost, and preserves the active
funnel run. Geometry, scalar reductions and target-status rows must not acquire
cross-row coupling. Shape-dependent random streams and shared load preclude a
paired path or speed-superiority claim. No numerical/public API changes are
authorized by the probe alone; implementation requires the explicit follow-up
contract above. The bounded diagnostic passes this audit.

### Native trial batching: preserve the existing independent streams

The capacity probe completed in 107.093 GPU seconds. All 18 calls passed native
health and exact repeated initial value/score/status checks. Warm L=25 times
per four-start bank were 1.386--1.430, 0.148 and 0.041--0.044 seconds for QR at
one, eight and 32 banks; nonlinear gave 0.677--0.695, 0.090 and 0.026--0.027.
They are descriptive shared-device measurements, with shape-dependent streams.
They justify testing a numerical batching mechanism, not adopting those raw
capacity outputs as trial evidence.

Implement an internal optional runner that batches **target evaluation** across
independent repetitions while keeping each repetition's original two-integer
TFP seed. Use the installed TFP sampling seed split, HMC momentum split,
SimpleLeapfrogIntegrator, kinetic-energy correction, safe log-ratio sum and MH
choice. Do not introduce a new integrator or scalar target loop. The source
anchors inspected before implementation are TFP `mcmc/sample.py:311,344`,
`mcmc/hmc.py:660`, and `mcmc/metropolis_hastings.py:175,202`, in the recorded
`tfgpu` environment. Copy no whole TFP module. Record hashes of these actual
dependencies for the diagnostic. Host instrumentation and metadata stay outside
the stable-signature XLA graph; pfor and NumPy remain forbidden.

For frozen start bank x, fixed kernel K and independent streams u_r, the intended
joint path law is the product over r of P_K(path_r | x). Flattening the row
dimension only in the batched value/score call preserves that product when the
target's declared rows are independent and each MH decision reduces over its
own parameter dimension. The runner must produce the same per-trial samples,
momenta, log ratios, acceptance bits and status as the existing sequential TFP
calls on the **same seeds**, including after permuting/grouping repetitions.
Do not infer that property from good acceptance. Any mismatch blocks integration
until explained and repaired. Shared uniforms, cross-trial reductions, wrong
first-step seed salts, missing rejections and loss of prefix health are specific
adversarial cases. Check Gaussian, QR LGSSM, nonlinear SSM and the supported
residual funnel map, with L=3 and L=25, CPU/XLA and trusted GPU/XLA.

Only a successful native check permits optional `replicated_trial_batch_size`
integration. Default 1 preserves old payloads and dispatch. Larger values bind
the execution identity, require exact-score batched mode and one complete
W+T path per chunk, and are bounded by the tested size. Before each grouped call,
durably charge every attempted trial using its existing seed/ordinal/cost;
persist its raw per-trial chunk and unchanged health/score after return. A lost
batch stays charged and retries the same streams; partial persistence must not
duplicate scores or change remaining streams. Changed grouping cannot change
the numerical sample, though measured cost metadata may differ. A native error
vetoes the whole attempted group, with no invented valid scores. Check these
boundaries in fresh-process recovery and all-member export/reload tests before
a fresh full-search price. Reject unsupported position-field/serial/partial-trial
configurations before numerical work, rather than silently falling back.

Reserve initially 1,800 CPU worker seconds and 600 GPU seconds for the native
implementation, numerical parity and diagnostics within existing release
ceilings. This is a bounded hypothesis. Freeze another source only after the
native gate passes; integration and complete pricing require their measured
remaining-budget allocation. No old source/checkpoint is edited or resumed
under changed identity. Six-L coverage, M100, W/T, original panels, every viable
candidate and statistical allocations remain unchanged.

Skeptical audit: a wider ordinary TFP call would change each trial stream, so
it is rejected as the implementation baseline. Instead compare exact existing
seed semantics using TFP's own integrator and arithmetic. Bound compilation
growth from per-trial random generation, check the two real-filter targets,
and preserve the row-product condition and all failed attempts. A native pass
is an engineering result, not full-search delivery, reliability or posterior
evidence. The staged implementation passes this audit; no confirmation starts
until complete pricing and its declared affordability test pass.

The first integrated CPU suite passed 23 checks, including both public routes,
six real-process lost-batch/partial-save recoveries and maximum-size parity on
all four models. The verified two-trial GPU suite also passed all 13 checks.
At size 32, however, first GPU calls for Gaussian and QR took 80.364 and 92.043
seconds: unrolling each trial's random-generation graph creates substantial
compilation cost. These are measured first-call costs, not steady-state timing.
The original maximum-size attempt keeps its cap and evidence.

Bound that graph growth by replacing only the per-trial random-generation and
seed-splitting unroll with native TensorFlow loops with one traced body. Target
evaluation and the TFP integrator remain batch-native; no pfor is introduced.
Require the same sequential-TFP exact trace parity at two and 32 trials, stable
tracing, and actual recovery before pricing. This is a graph-construction
repair, not a new random schedule. The existing source remains frozen. Extend
the native-repair GPU diagnostic allocation from 600 to at most 1,200 seconds
within the unchanged 60,000-second release ceiling to fund one corrected
maximum-size check; retain all prior costs. This convenience ceiling is based
on the observed compilation cost and does not promise affordability.

Skeptical follow-up audit: a loop can reorder random consumption or infer a
dynamic event shape. Keep the original constant `[chains, dimension]` and
`[chains]` random shapes and explicit stateless trial seeds; exact all-stream
parity is the veto. Shape/permutation tests and full raw-trial reconstruction
must pass. Compilation and wall limits remain in the accounting. The repair
passes this audit before execution.

The integration audit also checks attempted rows that have no saved trace:
every row in a partially persisted batch and every row in a failed native
batch must have a durable charge. Missing charges cannot be hidden by a valid
saved row or aggregate cost. Add explicit mutation controls for both cases.
Allow 900 additional CPU worker seconds for the final frozen-source accounting,
recovery and model-harness checks within the unchanged 18,000-second release
ceiling (native-repair allocation at most 2,700 CPU seconds total). This funds
actual multi-model recovery rather than replacing it with a mock-only test.

After native parity, final recovery/accounting and documented source assembly
pass, run **one** fresh QR full-search price on the repaired source with batch
size 32, the original `(20261002,2501)` development seed, all six L values,
M100 and the original repair/refinement/evidence settings. Use the existing
1,800-second search and 180-second closeout caps, reserving 2,000 GPU seconds.
The command is `scripts/run_hmc_v7_release_prices.py --source <frozen source>
--output <fresh root> --gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c
--cases lgssm_qr --wall-seconds 1800 --budget-seconds 2000 --trial-batch-size 32`.
The single model is a bottleneck diagnosis, not a reduced confirmation
inventory. Preserve the original nonlinear/funnel seeds for subsequent
three-model pricing. Inspect every stage time and require complete search,
all-member replay and checkpoint reconstruction. A timeout remains incomplete.
Native comparisons, failed older prices and exact base-seed spelling cannot
make source-dependent search trajectories paired.

Pre-price audit: this first complete-path attempt changes only the already
checked execution batching and health-transfer implementation, leaving the
statistical target and original scientific design intact. It can determine
whether the measured repairs permit full delivery and isolate closeout costs;
it cannot establish independent-seed reliability or forecast the other two
families. Unknown harness errors, mismatched sources/devices, missing charges,
invalid health evidence or the total cap stop this attempt. Ordinary candidate
failure triggers the next diagnosis. The bounded one-model price passes review,
conditional on the checks above; a successful launch is not its pass criterion.

October 3, 14:06 UTC audit outcome: the final RNG-loop implementation passes
17 CPU native checks and four trusted GPU maximum-size checks, with exact
sequential-TFP parity for all 32 streams on all four targets. Failed-batch and
partially saved batch missing-charge controls both pass. The initial assertion
failure compared tuples with their serialized list representation; normalizing
the test to JSON semantics fixes it without changing runtime behavior. Source-13
freezes the final charge audit. Its focused checks cover both public multi-member
routes, six real-process batch interruptions across Gaussian/QR/nonlinear,
legacy attempted-work checks, preparation/repair recovery and the complete-price
predicate. No additional native parity run is needed unless its runtime changes.
The conditional price still preserves every original scientific setting and
cannot authorize confirmation without complete delivery and affordability.

October 3, 14:18 UTC prelaunch result: source-13 assembled and built successfully
(44.005 CPU seconds). All 50 frozen-source checks passed in 663.560 enclosing
CPU seconds, including both public routes, all six batch recoveries, failed and
partial-batch attempted accounting, windowed preparation and both actual
repair/verification recoveries. This meets the stated prerequisites for the
single QR price. The final audit keeps the original seed, full design and
positive complete-search predicate; exact native parity is evidence about
streams, not about independent-search delivery. Proceed within the reserved
2,000 GPU seconds. Release remains pending.

The continuation audit found a harness trap before any subsequent family price:
development seeds were assigned by the index in `--cases`. Pricing nonlinear
or funnel alone would silently substitute QR's seed. Bind the three original
case names to their original seeds (2501/2502/2503 with first component
20261002), regardless of subset or order; reject duplicate/unknown cases.
Test the full, subset and reordered inventories. This preserves the original
scientific question and seed inventory, does not edit the live frozen runner,
and uses at most 60 CPU seconds inside the existing release balance. The
skeptical audit passes because it prevents seed substitution rather than
selecting successful seeds. A future runner records its own source hash while
continuing to validate the frozen numerical tree.

### Conditional host profile after the batch32 price

The live price's immutable records cover 2,752 completed chunks and 86.117
seconds in native calls plus immediate serialization, while enclosing search
time is much larger. These buckets are descriptive and do not establish a
paired speedup. Source inspection also finds full-history rehashing at internal
work boundaries and rehashing of growing partial chunks at each durable save.
Keep the live attempt unchanged through its cap. If complete-work cost still
prevents confirmation, run a bounded writer profile on its saved checkpoint,
not another numerical search.

Question: which checkpoint-writing operations consume repeated host work?
Comparator: full and incremental writes of the same saved source-13 state.
Read its immutable evidence, verify ordinary hashes and reconstruct only the
controller state; a minimal writer state carries those decoded records without
claiming numerical replay. Copy all referenced records to a new diagnostic
directory before writes. Profile the actual writer three times in each mode,
preserve timings, call counts and cProfile output, and cap the whole diagnostic
at 180 CPU worker seconds within the release ceiling. No GPU initialization,
new samples, changed policies or admission result is produced. Hash mismatch
or an invalid controller snapshot stops the diagnostic. The saved snapshot
and source manifest identify the exact workload.

Skeptical audit: a fabricated tiny checkpoint would not reproduce growth with
history; use the actual saved state and original immutable records. Profiling
overhead makes timings descriptive. Do not profile concurrently with the live
price or infer numerical validity from a writer-only fixture. Any subsequent
optimization must retain charge-before-call, recovery of lost and partial
results, full explicit/final/export/reload validation and unchanged numerical
streams. This conditional profile passes the audit before execution.

The price ended incomplete after 1,980.571 GPU seconds. It exported all seven
verified members by 1,955.855 seconds, but reconstruction/reload did not finish.
The writer profile completed in 24.941 CPU seconds on the actual 100-record
checkpoint. Full saves took 4.025/4.807/4.501 seconds; incremental saves took
0.139/0.138/0.138 seconds, with identical written bytes. Full saves spent
13.077 of 13.333 profiled seconds hashing/serializing accumulated evidence.
Incremental saves still cost about .14 seconds each at this state, and the
batch driver invokes them twice per trial. This is a concrete engineering
repair trigger, not a changed numerical policy or a successful full price.

Apply these two bounded changes together. During the repository-owned
controller's internal dispatch, use incremental saves of unchanged persisted
history. Force a full write when dispatch returns or raises, including an
incomplete/budget/paused result, and retain full explicit-save/export/reload
validation. For native trial batches, group the individual charge events into
one durable checkpoint before the native call, then checkpoint the returned
rows together. If row serialization raises halfway, flush the completed prefix
and every prior charge before propagating; recovery must preserve those rows
and rerun only the unsaved streams. Individual trial records and charges keep
their current formats. Ordinary single-trial execution is unchanged.

The skeptical audit requires mutation controls for history changed between
work items, both normal and exceptional terminal paths, and explicit saves
after incremental writes. A grouped-charge failure must flush the charged
prefix without a native call. A lost native batch remains charged. Test a
failure while serializing the second returned row to exercise a real partial
prefix under the new save cadence; do not manufacture a passing recovery test
by removing that case. Reuse checked native math/parity because no RNG,
integrator, health threshold or trial horizon changes. Rerun affected actual
process recovery, all-member and budget/integrity tests on a fresh source.
Reserve 1,600 CPU worker seconds within the existing balance for implementation
checks, fresh assembly and documentation. Any further numerical price needs
passed checks and its own remaining-budget reservation.

In parallel with this host repair, profile one actual grouped reload of the
seven saved source-13 members, using the existing `profile_group_gpu_reload.py`
on the same trusted GPU with verified growth. The 330-second profile alarm and
350-second subprocess cap fit a 360-GPU-second reservation. Preserve the
snapshot, source manifest, member inventory, pstats, receipt and failure state.
The primary check is exact equality of reloaded and verified IDs; a timeout
provides only a partial cost profile. Profiling overhead makes its timings
descriptive, and replay cannot establish full search or independent reliability.
This targets the unmeasured closeout cost without a new numerical search and
passes the pre-execution audit. Frozen source-13 and all failed attempts remain
unchanged; terminal release and confirmation remain pending.

October 3 follow-up: an initial grouped-save implementation restored the
incremental flag before flushing, accidentally making every grouped save a
full-history check. This explains the single-trial repair-test child timeouts;
the failed/incomplete suites are retained and charged. Restore the flag only
after the durable flush, including when the flush raises. The corrected
boundary suite passes all 25 cases, including explicit mutation checks,
normal/exceptional dispatch exits, charge-before-call and serialization-prefix
durability. The first exploratory suite also edited test source while running;
it supplies no full-source claim and receives a conservative 1,000-second
charge. Raise the host-repair phase allowance from 1,600 to 2,800 CPU seconds
inside the unchanged 18,000-second release ceiling. Use the remaining focused
checks on a frozen tree; do not repeat unchanged native parity or change
scientific settings to address a harness timeout.

The separate source-13 GPU profile reloaded all seven exported members and
matched their verified IDs (271.859 enclosing seconds, 267.219 profiled seconds).
The prior export cost was 148.176 seconds. Therefore the inherited 180-second
closeout allowance does not price a complete handoff at this evidence size.
After source-14 checks pass, allow one original-seed QR full-search price with
the unchanged 1,800-second search cap and a separately recorded 900-second
closeout cap, reserving 2,800 GPU seconds. The 900 seconds are a convenience
upper bound informed by measured export/reload cost plus still-unmeasured
checkpoint reconstruction and accounting; they are not a runtime guarantee.
Expose the closeout cap in the pricing runner and its manifest, keeping the
historical 180-second CLI default. Test that budget validation includes it.
No new trials are authorized after the search cap, and longer closeout cannot
turn unfinished search work into complete delivery. Preserve every earlier
failure. The audit passes for this diagnostic price, not confirmation.

### October 3 continuation: remove repeated seed derivation before repricing

Source-15 reaches its search cap with 14 verified candidates and eight pending
work items (three initial measurements and five verification extensions).
Its closeout is still running. Preserve that attempt; even successful replay
cannot make its unfinished search complete. The source-13 replay profile is
still applicable to the unchanged seed validator: 313,224 `work_seed` calls
take 34.757 profiled seconds. Both chunk validation and charge reconstruction
derive the same work root inside each row's entire batch inventory. For B
rows this repeats B seed-list derivations of length B. This is unnecessary
host work, not a lack of independent numerical trials.

The next bounded repair hoists a work's fixed root once per validation call,
reuses each expected batch seed list within that call, and checks each complete
batch's charged membership once. Every row still has its identity, metadata
and individual charge checked; no cache survives the call. A single-chunk trial
also needs no TensorFlow concatenation before analysis, while multi-chunk
continuity remains unchanged. Do not alter RNG splitting, hashes, arithmetic,
health decisions, evidence horizons, candidate budgets or durable formats.

Engineering criterion: outputs and reconstructed seeds equal the frozen
source-15 behavior; existing rehashed seed/charge corruption controls still
fail, including an unsaved batch row and a classified failed batch. Test at
batch sizes 1, 8 and 32, altered first and last row metadata, reordered charges,
and an additional validation after mutation. Preserve multi-chunk and both
public-route coverage. Timings and call counts are explanatory diagnostics;
they cannot establish full-search completion, affordability or release status.
Use at most 350 CPU worker seconds from the remaining 716.509-second release
balance, including focused tests and source assembly if those tests pass.
This convenience debugging ceiling leaves the existing 250-second closeout
reservation intact. No new numerical price is authorized by this subsection;
first settle source-15 and calculate the remaining budget and required cost
reduction for independent confirmation.

Skeptical audit: caching decisions across replay boundaries would conceal live
mutations and is rejected. A work root and batch membership are pure functions
of fixed inputs within one synchronous call, so local reuse preserves the
checked object. Corrupted row metadata must still be compared for every row;
checking only the first row would be wrong. Unknown work IDs, missing charges,
out-of-range ordinals and malformed seeds remain errors. The current GPU run
uses its frozen source and cannot be modified by this repair. The plan passes
for this localized implementation/check step; complete prices and the 32-seed
proposal remain unresolved, and no statistical or timing superiority is sought.

The price is now terminal: 2,701.022 enclosing GPU seconds, all 14 verified
members exported, eight search work items pending, and no complete reload or
checkpoint/accounting result before the process ceiling. Export alone took
783.942 seconds after tuning returned. The GPU balance is 36,334.228 seconds;
dividing it by the proposed 96 confirmation searches leaves at most 378.482
seconds per search before any further pricing or final checks. This is a
derived affordability requirement, not a measured runtime. Repeating the
current full price or extending its timeout cannot by itself satisfy it.

The seed repair passed 17 existing checks and nine added controls after a
fixture omitted required policy fields and was corrected. The frozen source-16
suite then hit its enclosing 180-second cap without a terminal report. Preserve
the failed and incomplete receipts. Finish the outstanding regressions in
separate bounded processes and compare old/new charge registries on the actual
saved 9,056-charge checkpoint. These reference checks use no new HMC samples.

To fund that completion and the next measured host-analysis repair, increase
the local CPU release allocation from 18,000 to 21,600 worker seconds. The
additional 3,600 seconds come from the previously authorized October 2 CPU
grant: its reconciled 70,536.184-second balance preceded this release allocation,
so 48,936.184 seconds remain unallocated after this amendment. Do not change
the GPU ceiling or draw from the old-policy GPU campaign. This is a bounded
reallocation within the existing owner budget, not additional compute authority.
No rerun may silently reset a failed attempt's charge.

Call-chain audit correction: the exact-score binding already caches an analysis
by the current raw-evidence hash, including predecessor calls. The presence of
recursive predecessor code alone therefore does **not** prove duplicated
numerical reconstruction. Do not implement another cache based on that reading.
Measure the current reconstruction cost before changing its health/score
execution. Any batched diagnostic repair must preserve each trial's existing
payload and failures, with first/last-row corruptions, mixed healthy/invalid
trials and chunked paths checked against the frozen implementation. No new
full-search price until the remaining cost has a measured explanation and a
plausible path toward the derived confirmation requirement.

The saved-charge comparison passes on all 9,056 attempted streams: old/new
registries are identical. Its single descriptive measurements are 20.344 and
.133 seconds. This removes that repeated work, but cannot explain the whole
783.942-second export. The saved export bundle is 2.013 GB; raw work records
repeat cumulative trials and chunk payloads. Those bytes remain preserved.
Before a broader format or analysis repair, profile one complete saved work
record's 64 new trials on the bound GPU, using source-16's existing trial
assembler and the original raw inputs. Require identical reconstructed trial
payloads to the saved record; no target calls or HMC transitions are allowed.
Cap this diagnostic at 180 enclosing GPU seconds inside the existing balance,
with a 150-second child cap. Record source/raw hashes, growth, device, profile
and all failures. This is a measured attribution test, not a full replay,
speed comparison or numerical admission. Its pre-execution audit passes:
the work and comparator are exact, the raw record is immutable, and a mismatch
vetoes the proposed shortcut rather than selecting a different record.

The 64-trial reconstruction passed with exactly equal saved payloads in 5.586
profiled seconds (9.483 enclosing GPU seconds). It issued 6,592 eager fast-path
operations plus 192 graph calls. The measured Boolean AND/OR/inversion calls
come from per-trial target-status validation. Fuse each status check into one
stable-signature TensorFlow/XLA reduction, retaining host shape/dtype/schema
validation and the exact malformed-input errors. Similarly combine the three
retained finite-value reductions into one Boolean vector. These operations
contain no floating-point sums or changed thresholds. Preserve original input
dtypes, especially unsigned integer status/floor values and complex optional
conditioning fields, and handle absent conditioning fields explicitly.

Audit before implementation: casting every integer to signed int64 would
misclassify large unsigned floor counts and is rejected. A shared status code
must not hide failure in another trial; each invocation checks its original
trajectory, with no cross-trial pooling. Existing target-status and native
health hostile fixtures plus direct integer/complex checks compare exact
Boolean outputs and full trial payloads with source-16. Evaluate XLA compatibility,
stable tracing and GPU memory growth before adopting the fused kernels. Allow
600 CPU worker seconds and 180 GPU seconds within the amended release balance
for focused tests and a repeat of the same saved-record diagnostic. This
localized repair has no new HMC sampling or statistical promotion authority;
full-search repricing still waits on an adequate measured cost explanation.

Fused validity checks pass 136 CPU tests and 46 trusted GPU tests plus 32
saved-trace comparisons; all payloads remain identical. The same 64-trial
assembler also reproduces every saved payload after the repair. Its descriptive
time remains 5.528 seconds, so these small fusions do not establish a sufficient
cost reduction. Next, test one isolated batched **summary** graph on those same
64 trials. It maps the existing pure summary function with a native TensorFlow
loop and preserves each trial's four-chain axes. This is not a row-mapped
target evaluation, pfor, or a new tuning implementation. The outer graph remains
non-XLA in this diagnostic to preserve the existing summary's exact arithmetic;
HMC remains untouched and no default path is promoted from this exception.
Compare every tensor with the original per-trial graph before reporting timing.
Measure one cold plus two alternating warm calls at batch sizes 1, 8 and 32;
these fixed repetition counts are engineering probes with descriptive timing
only. Bound the experiment at 120 GPU seconds within the existing balance.
If exact parity fails or call cost remains inadequate, reject this shortcut and
retain the failed diagnostic. A favorable result requires a separate caller-path
implementation and hostile-trace/recovery checks before any full price.

The following proposed storage repair was **withdrawn by the post-implementation
audit**. File size did not establish write-time attribution, and the unmeasured
format added portability dependencies. Source-18 and its test receipts remain
historical; active runtime and guide retain the source-17 v1 bundle. No release
gate is closed by that attempt.

A normal release export
already has every immutable JSON evidence file under
`tuning/numerical_evidence/`; copying those same bytes into a second v1 bundle
is unnecessary. The retained loader now supports a v2 manifest bundle that
records each relative path and digest, validates the raw canonical JSON bytes,
then parses and runs the existing full evidence/member checks. Missing files,
path traversal, malformed rows and digest changes fail closed. Generic callers
without a completed checkpoint still receive the v1 self-contained bundle.
This is a storage representation change with no authority, threshold, seed or
evidence deletion. The focused manifest test passes, while the existing v1
portable/member tests remain unchanged. A fresh source assembly and the full
grouped I/O suite are required before another numerical price.

The summary-batch probe also failed its stated evidence contract: attempt 1
compared an invented non-XLA reduction to the existing XLA scores and failed
exact equality. Attempt 2 changed the comparator and allowed a tolerance while
retaining an incorrect `exact_parity` label. Both remain diagnostic-only and
are retracted as adoption evidence; the second pass cannot cure the first.
Use a new explicit probe that calls the **actual unchanged** summary and score
graphs from source-17 inside a TensorFlow native loop. No implementation copy,
reduction rewrite, altered comparator or tolerance is permitted. Check every
returned tensor exactly, including all four measured score windows and health
summary fields. Evaluate one cold and two alternating warm calls at B=1/8/32,
record both scalar and grouped times, and bound at 120 GPU seconds from the
existing allocation. These calls consume the same first 64 saved trials and
cannot issue numerical evidence. The non-XLA outer orchestration graph is an
explicit diagnostic exception that preserves the existing inner graph choices;
it is not a claim of XLA compilation of the whole calculation. This revised
probe passes skeptical review because it tests the actual arithmetic directly
and has an exact no-change criterion before any caller integration.


## October 4: whole-cost audit and bounded placement diagnostic

The release budget and all late source-16/17/18 receipts are reconciled before
further work. The proposed 96 searches can consume at most 377.908 seconds each
from the present balance even before pricing and closeout reservations. Earlier
incomplete runs do not provide that price. No new full search is authorized by
these microbenchmarks alone; require a plausible measured cost explanation.

The actual call chain is `observe -> _assemble_trials -> analyze_trial` and
`complete_trial_scores`; grouped export/reload calls `_validate_member_set ->
evidence_analysis -> _assemble_trials` on uncached records. Existing digest-keyed
analysis reuse already prevents repeated predecessor reconstruction within a
binding. Original emission does not populate that cache; this observation alone
does not justify bypassing reconstruction. Profiling reports 384 graph calls
and 5,184 eager fast-path operations for 64 saved trial reconstructions. The
existing summary helper describes CPU checkpoint arithmetic, but the wrapper
has no explicit placement; current GPU-scope diagnostics put it on the GPU.
Placement is a hypothesis to test, not an established cause.

**Question and comparator.** On the same 64 original source-15 QR trials, does
source-17's unchanged complete trial assembler produce exactly the same saved
payloads when run on CPU and GPU, and where is its time spent? Call the actual
functions, not copies. Decode the same payloads and compare the full returned
JSON, including every health field, score, window and raw tensor. First run one
cold call per device, then one warm call in reversed order. These are diagnostic
repetitions chosen to expose tracing/transfer costs, not independent timing
replications. Add component timing around original functions without changing
outputs. No HMC transitions or numerical replay authority are issued.

**Primary pass condition.** Exact payload equality against the preserved record
and between placements. Any discrepancy rejects a placement change; do not
introduce tolerances or change the comparator after seeing results. A timing
change is descriptive only and cannot close affordability or release gates.
Device readiness/memory-growth failure invalidates this diagnostic. Preserve
its failure rather than substitute a different seed/record.

**Budget and execution.** At most 120 enclosing GPU seconds from the existing
allocation (GPU is visible throughout, including host comparison); child cap
110 seconds. Use tfgpu, four pinned CPU cores, bounded threads, trusted GPU
access and verified memory growth. Preserve script/source/raw hashes and exact
command in a fresh `placement-profile-01` directory. The CPU computation is an
explicit small reference diagnostic, not a proposed production backend. The
probe script is `artifacts/hmc-v7-release-2026-10-02/probe_trial_placement.py`.
The original source-15 record and source-17 tree are immutable comparators.

**Skeptical audit.** Wrong-source replay cannot issue authority; this unbound
saved-data diagnostic therefore makes no replay claim. Exact tests prevent
cross-device rounding from silently changing acceptance decisions. Full-graph
profiling must distinguish cold compilation, graph dispatch, serialization,
health and statistics; cumulative times cannot be added as independent costs.
The observed file size is not a storage bottleneck measurement. Budget caps are
convenience stop limits, not evidence of adequate search time. This diagnostic
passes pre-execution review for attribution only. If it fails parity, preserve
that result and investigate only the independently equal components; do not
adopt whole-assembler CPU placement.

After the diagnostic, record the repair/no-repair decision and its evidence.
Any adopted change needs focused mutation, partial-recovery, multi-member and
model-specific integration checks plus a fresh source assembly. Only then,
if the cost forecast is plausible, run a bounded complete full-grid price with
original panels/seeds. Freeze confirmation only after all three families have
complete prices; final scope/default/guide decisions remain P5.


### Placement result and complete saved-replay attribution

`placement-profile-01` ended normally with a parity failure (11.208 enclosing
GPU seconds). GPU reconstruction matched all 64 original trial payloads; CPU
changed health summaries, normalized displacement, start scores and temporal
windows. The CPU placement proposal is rejected. Warm times were 1.109 seconds
on GPU (health .902, scores .043, decoding .089, encoding .052) and .577 on CPU.
These are descriptive component measurements; no tolerance is introduced.

The next discriminating measurement is **complete actual saved-member replay**,
using source-15 and its own original `gpu-price-batched-02` evidence. Call the
public grouped loader on all 14 exported members with the bound QR target;
require exact verified-ID set equality and all existing identity, checksum,
reconstruction, membership and failed-attempt validation. No HMC samples, new
candidates or scientific promotion are produced. Instrument actual loader,
validation, analysis, trial assembly, JSON and persistence functions to report
call counts and inclusive times, plus a full profile. Do not infer work from
nested cumulative times by adding them.

Allocate at most 900 GPU seconds within the remaining release budget, with
840-second worker and 870-second enclosing caps. The earlier seven-member
source-13 cold replay took 271.859 seconds; the larger 14-member source-15
export took 783.942. These measured baselines motivate a bounded attempt,
not a guaranteed finish. Preserve incomplete profile data on deadline. The
artifact directory is `full-replay-profile-01`; the exact copied diagnostic
script and command go in its manifest. Original source/data remain unchanged.
If it completes, it closes only saved-member reload for the incomplete search.
If it times out, the profile is still attribution evidence, never replay success.

Skeptical audit: a latest-source binding cannot authenticate older evidence, so
this run intentionally uses the evidence's frozen source-15. Later seed/health
repairs are accounted for separately and cannot be credited as measured whole
savings. The actual public loader, original target, complete member inventory,
all failures, memory-growth verification and enclosing cap answer the remaining
cost question without changing the method or admission policy. This audit
passes for one attribution run; it does not authorize another full search or
confirmation. A failed placement probe does not invalidate this distinct test.


### Independent repair: reuse the already computed emitted analysis

The placement profile attributes .902 of 1.109 warm seconds to health analysis
for 64 saved trials. The inspected emission path already calls the same
`_assemble_trials` and `_analyze_trials` used by reconstruction, but does not
populate the existing content-digest analysis cache. Thus the next predecessor
or export request repeats those numerical calculations. The original-source
full replay profile continues independently while this localized repair is
checked; it cannot be modified by edits to the active tree.

Use the **existing** digest cache for a newly emitted successful record only
after validating the entire completed chunk inventory and using its decoded
values to assemble trials. The cache stores a deep JSON copy of the computed
analysis under the finished numerical-record digest. Every subsequent request
still hashes current content; external reload starts with an empty cache and
reconstructs all records. Failed-execution behavior remains unchanged. This
adds no persistent cache or new trusted input. Existing binding/source checks,
all-record digests, receipt validation and all charged-seed checks stay active.

The primary correctness criterion is exact equality between the emitted cached
analysis and a forced fresh numerical reconstruction for single/chunked and
batched execution. Count assembler invocations to demonstrate the removed
recomputation directly. Required negative controls: changed raw samples, scores,
work metadata, seeds and rungs; missing charges for an unsaved row; corrupt
new native row metadata must fail before an observation/cache entry is issued;
returned analysis mutation cannot poison later reads. Exercise both public
multi-member export/reload routes and actual QR/nonlinear partial-process
recovery using their existing tests. Do not rewrite an expected result to make
cache parity pass. No arithmetic, horizon, seed derivation or admission change.

Skeptical audit: caching the analysis before checking native row metadata would
skip a check that previously occurred at replay and is rejected. Reusing decoded
validated chunks avoids an extra parse while preserving that check. Caller
mutations produce a different current digest; rehashed malformed records must
still hit reconstruction and fail. A fresh binding must recompute, never import
this cache from a serialized artifact. These controls address correctness and
the exact call chain, not runtime superiority. The focused implementation passes
this pre-execution audit. Reserve at most 600 CPU worker seconds for focused
checks and source assembly within the existing balance. A full-search price
still requires the completed cost audit; cached in-memory export cannot by
itself certify independent reload or confirmation affordability.


The saved replay passed for all 14 IDs at 510.927 enclosing GPU seconds.
Measured inclusive costs: assembly 298.163, health within assembly 231.721,
chunk validation 77.116, registry construction 42.106, file parsing/checksums
22.395 seconds. The profile is original source-15; do not subtract these nested
times as independent savings or call it a current-source price.

The emitted-analysis suite initially passed 40 checks and failed two: one new
test used a nonexistent method, and the shared position-field mechanics runtime
has no `_analysis_cache`. The corrected implementation reuses only an existing
cache; the mechanics branch keeps its reconstruction behavior. The two focused
reruns pass; source-19 assembled and 44 focused frozen checks pass. Actual QR
and nonlinear process-recovery checks are in progress. Failed attempts remain
charged. These tests establish equivalence/rejection behavior, not complete
search affordability. The separate old-policy SSM campaign still reports active
work (its status refreshed October 3, 20:15 UTC); its reservation is not freed.


### Source-19 full-procedure price after the measured reuse repair

All 44 focused source-19 checks and both real QR/nonlinear partial-row process
recoveries pass (74.114 and 193.245 CPU seconds). The original nine new controls
are included in that frozen suite. The emitted-reuse repair phase consumed
413.930 CPU seconds including its failed attempt, rerun and source build,
below the 600-second ceiling. Source-19 manifest SHA-256 is
`da6609ba97030ef5765b7c2592d7004bd8bebd5e1e7d1948542197f46e7b390e`. Source-18 remains ineligible.

The whole-replay profile supplies the missing cost attribution. Existing
seed/validation reuse and the now-tested emission reuse remove identified
repeated calculations. A **single** complete QR price can now answer whether
the repaired procedure finishes, and measure its remaining cost. It cannot
pre-certify the proposed 96-run confirmation's affordability. This is the first
price containing both those repairs; no three-family sweep is launched yet.

Use the original QR base seed `(20261002,2501)`, six-L primary grid, L=4/7
refinement, M100, W=3/T=65, 32--256 repetition schedule, epsilon hypotheses,
all thresholds and every verified member. A new source changes scope/derived
streams; equal base seeds do not make this a paired performance experiment.
Give search its existing 1,800-second cap. Increase the separate closeout cap
to 1,200 seconds: twice the observed 505-second cold replay plus approximately
190 seconds for export/accounting are a conservative **planning hypothesis**,
not measured current-source cost or a guarantee. The enclosing reservation is
3,100 GPU seconds, covering the 3,000-second process cap and launcher overhead.
This changes resource allowance only, never an admission threshold or denominator.

Exact command (tfgpu, trusted GPU, memory growth and per-chunk device/XLA
checks remain mandatory):

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_hmc_v7_release_prices.py --source docs/plans/artifacts/hmc-v7-release-2026-10-02/source-19/source --output docs/plans/artifacts/hmc-v7-release-2026-10-02/gpu-price-emission-01 --gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c --cases lgssm_qr --trial-batch-size 32 --wall-seconds 1800 --closeout-seconds 1200 --budget-seconds 3100
```

Skeptical review passes for this bounded development question: original design
is preserved, real-model recovery is checked, a measured repeated calculation
is removed, the output is new, and complete delivery/reload/reconstruction is
the pass criterion. Timeout or partial delivery remains a failed price and a
repair trigger. A complete but expensive result closes only its functional
case; it does not close reliability or justify spending the confirmation budget.
Record exact complete prices before any denominator/funding decision. Shared
SSM work remains separately reserved and cannot validate v7.


## Confirmation analysis audit while the checked price runs

The prior goal turn made progress: it changed the implementation, passed 46
frozen-source checks and started one bounded price. The existing process handle
was polled and remains active; this is not permission to duplicate it.

The plan's denominator means **every predeclared independent slot**. A timeout,
resource deferral, empty delivery or inconclusive result contributes zero
successes. Report numerical and resource dispositions separately. Every slot
needs a disposition and every attempted enclosing cost must be charged; a
successful slot requires the complete procedure, all-member replay/checkpoint
checks and its stated cap. The binomial bound is about potential complete-work
delivery over independent seeds, as derived above; it is not a homogeneous
machine-load deadline probability. The requirement does not allow missing slots
to disappear or partial members to become successful complete searches.

Implement a small **diagnostic analysis** module and CLI, with no sampler or
launch authority. It must (a) require complete prices for all three declared
families from one frozen source/device before reporting a full point forecast,
(b) retain the original independent-slot denominator and reject duplicate or
unexpected slot IDs, and (c) compute the simultaneous one-sided lower bounds.
Missing/incomplete prices produce an explicit unpriced result; forecasts remain
descriptive and cannot authorize a launch. This helper does not freeze a design,
select seeds, change the 32-per-family proposal or mark a release ready.

For exact decision arithmetic, at required delivery p0=4/5, compare
`sum(comb(n,k)*p0**k*(1-p0)**(n-k), k=s..n)` with alpha/3=1/60 using rational
integers. Monotonicity of the binomial upper tail implies that the one-sided
lower confidence bound exceeds p0 exactly when that tail is below 1/60.
For a displayed lower endpoint, bracket its root using rational bisection for
64 iterations (11 bits beyond binary64's 53-bit significand), then round toward
zero. This is an exact binomial-tail calculation for the declared finite n;
it is not an IID claim about wall-time success under arbitrary contention.

Skeptical audit: using a two-sided .05 interval without changing its tail
allocation would answer the wrong confidence question. Treating only completed
slots as n would condition on resource/sampler success. Adding one-family
forecasts or using old-policy/source prices would answer the wrong cost
question. The helper must reject these paths. Cross-check every s=0..32 against
an independent SciPy beta reference, plus zero/all-success analytic cases and
exact rational comparisons at p0. SciPy is used only in the test reference;
implementation uses Python standard-library arithmetic. Tests must include
an omitted family, duplicate slot, failed/missing slot, partial member delivery,
mismatched source/device, nonfinite costs and caller-edited success flags.

Allow at most 180 CPU worker seconds for the focused implementation checks,
inside the current release balance. No GPU work or publication is involved.
Artifacts go in `confirmation-analysis-checks-01`; implementation and tests are
added to the next source assembly if needed, not to the running source-19 tree.
The live source, target, thresholds and denominator proposal are unchanged.
This audit passes for engineering/statistical reporting, with no release or
confirmation-authorization claim.


## Source-19 complete price and next cost attribution

The full QR price passed in 2,318.380 enclosing GPU seconds. It completed 42
candidates (22 verified, 20 promotion-failed), all-member export/reload,
checkpoint reconstruction and attempted-work accounting. All 12,096 trials were
complete and valid; 3,290,112 transitions and 40,282,112 gradient-work units were
charged, with no attempted work outside complete trials. Stage endpoints were
1,103.957 seconds for tuning, 1,169.701 for export, 1,746.402 for reload,
2,297.883 for checkpoint reconstruction and 2,308.241 for accounting.

The launcher enforces the **combined** search-plus-closeout process cap, not
a separate stopwatch starting at tuning return. Thus unused search allowance
can fund closeout. Here closeout took 1,204.284 seconds, while the total was
below the 3,000-second process and 3,100-second enclosing reservations. Earlier
phrasing implying an independent 1,200-second closeout kill was imprecise.
This clarification preserves the actual enforced cap and all measured costs.

The 32-seed QR point forecast alone is 74,188.171 seconds, exceeding the entire
remaining 33,438.688-second release GPU balance before either other family.
There is no affordable confirmation design established. Keep all three families,
full L coverage, horizons, thresholds and planned-denominator discipline. The
new analysis helper passed 82 combined confirmation/pricing checks; its CLI
correctly refuses a complete forecast from the older incomplete QR price.

Before further numerical prices, read the completed source-19 raw chunk files
once and sum their existing timing fields. Verify each canonical JSON digest,
unique work/trial/chunk/seed identities, completed-work membership and exact
12,096-chunk count against the settled model/checkpoint. Preserve the metadata
scope `equal_share_of_enclosing_batch_call` and sum these per-row shares once,
never multiply them by batch size. Separately sum `elapsed_seconds`, which
includes the surrounding native call share and per-row tensor serialization.
These are host-observed spans, not a hardware counter or synchronized isolated
GPU-kernel benchmark. Compare the totals with enclosing search and closeout;
report uninstrumented time explicitly rather than assign it speculatively.

This is a read-only CPU cost diagnostic, with GPU hidden before any framework
import (none is needed), at most 120 CPU worker seconds in the current balance.
Output: `source19-cost-accounting-01`, with script, command, source/configuration
hashes, counts, per-stage/per-L descriptive timings and a terminal receipt. No
HMC, retuning, acceptance decisions or posterior claims are produced. The
pre-execution audit passes: the complete source-19 baseline is fixed, nested
costs cannot be double counted, mixed/unknown timing scope is rejected, and
missing/corrupt evidence invalidates the report. A shared-GPU observation during
closeout found two TensorFlow processes and 86% total utilization; that point
alone cannot attribute delay to either process or establish a timing ranking.
The next repair must address the measured remaining cost, not relaxed evidence.

## Bounded complete-health batching investigation

Source-19 is the complete-procedure baseline. The earlier two-summary batching
probe saved only a fraction of total assembly cost, so it does not justify a
runtime change. Inspecting the actual assembler shows additional per-trial
native validity, retained validity, two target-status graphs, tensor decoding
and host transfers. The next discriminating diagnostic groups **these existing
operations**, without changing their arithmetic or existing device placement.

Question: can a stable TensorFlow loop execute the complete existing set of
per-trial numerical reductions with exact outputs at materially lower overhead?
Use saved source-19 QR chunks, comparing every returned tensor to the original
scalar functions on the same tensors. Include input preparation and host
materialization in reported costs; separate first compilation from warm calls.
No target calls, HMC sampling, new seeds, admission or release are involved.
A mismatch, missing reduction, source change or GPU allocation-policy failure
invalidates the proposed repair. Slow execution rejects its cost hypothesis;
it does not invalidate v7 or the completed QR evidence.

Defaults and assumptions: group sizes 1/8/32 come from the earlier diagnostic
and existing maximum native batch; they are diagnostic hypotheses, not defaults.
Three alternating-order repetitions give descriptive timings only, not a
statistical runtime ranking. A 120-second process cap is a convenience bound
for this saved-data probe; reserve 130 enclosing GPU seconds. The outer graph
may remain non-XLA because it composes existing mixed-XLA health/score functions
and CPU string decoding. That is diagnostic graph orchestration, not permission
to change the XLA HMC kernel or enable pfor. No eager/CPU placement substitution
may be labeled exact-equivalent without a checked comparison.

If the complete reduction probe passes exact parity and reveals useful cost
savings, implement a private batch analysis path with explicit shape/dtype and
status validation, all existing invalidity precedence, and scalar comparison
controls for invalid rows. Preserve the scalar path for heterogeneous/chunked
layouts that the batch implementation does not support. Test frozen starts,
discarded-prefix failures, NaN/Inf, MH mismatch, movement/resonance, optional
status fields and divergence, bytewise raw-evidence reconstruction, and both
public routes before any fresh full-search price. No unvalidated reduction or
persistent trust cache can supply an acceptance decision. Allow at most 900
CPU worker seconds for focused implementation/regression and source assembly
in this phase; a further GPU check needs a recorded allowance before launch.

Skeptical audit: this probe compares actual existing functions, preserving the
fixed source and original saved inputs; timing includes transfer/packing costs
and does not count overlapping measures twice. Exact tensor equality is a
necessary engineering criterion, never a substitute for full payload parity,
invalid-input tests or complete family pricing. No reliability denominator,
scientific threshold or release requirement changes. Audit passes for the
bounded diagnostic, conditional implementation and focused checks only.
Artifacts: `complete-health-batch-probe-01` and new numbered focused-check roots.

The first complete-reduction probe passed exact equality on all 32 saved QR
trials in 19.472 enclosing GPU seconds. Warm scalar calls cost .56--.69 seconds;
groups of 8/32 cost .38--.45, including packing and materialization. This modest
descriptive saving does not justify a broad runtime refactor or establish
confirmation affordability. No runtime change has been adopted.

One further diagnostic within the same 130-second GPU phase allowance tests
chain-independent health summaries. The existing summary's movement, recurrence,
block means and realized acceptance vectors operate independently along each
chain; concatenate trial/start columns for that existing computation, then
reshape them back to [trial, start]. Recompute pooled means, realized means and
signed extrema separately within each original trial. V7 trial scores still
come from the existing scalar XLA function in a bounded loop. No trial/start
column is pooled with another trial for a statistical decision. This is an
algebraic batching hypothesis, not an established floating-point equivalence:
every scalar, vector and matrix must equal the original 32 per-trial results
exactly before it can be considered. Preserve the previous loop-only result.
Stop this direction on any mismatch; do not introduce tolerances to rescue it.

### Revised repair after the batching comparison

The chain-concatenation hypothesis **failed exact equality** in the normalized
return vector on the original saved QR traces. Its 9.162 GPU seconds and failure
remain recorded in `complete-health-batch-probe-02`. No tolerance, scalar summary
replacement or runtime batching change is adopted. The first loop-only probe's
small savings do not resolve the cost gate. This rules out those two proposals
as the next runtime repair; it is not evidence against the tuning method.

The inspected closeout call chain offers a smaller exact-preserving repair.
The model harness independently loads every exported member into a new binding,
then loads the same numerical evidence into another new binding for checkpoint
reconstruction. Both readers check their own files, hashes, target/source/device,
work identities, observations and endpoints, but repeat deterministic raw-trial
analysis. The measured second stage cost 551.481 seconds in source-19.

Within **one synchronous fresh closeout operation**, allow the two new reader
bindings to share analyses keyed by execution identity, complete evidence-rung
schedule and a freshly computed digest of current raw evidence. A private scoped
cache starts empty; live tuning bindings cannot seed it through their local
cache, and only actual raw reconstruction populates it. Every reader still
checks all files, identities, predecessor hashes and accounting. A missing or
changed digest triggers reconstruction or the existing rejection. The cache
ends with the operation, including exception exits; later independent reads
remain cold. This changes no numeric computation, persisted format, tuner entry
point or release predicate. Independent replay means independence from the live
tuning cache, not repeating identical arithmetic twice within the same checked
closeout.

Use the existing 900 CPU-second implementation/check reservation. Required
controls: two standalone cold readers versus scoped combined closeout produce
identical candidate states, members and endpoints; scoped closeout reconstructs
each distinct numerical record once; rehashed/raw/checksum corruption in either
file family still fails; duplicate/unknown members, missing attempts, changed
source/target, different bindings or rung schedules cannot share trust; scope
exit, exception and nested scopes do not retain a shared cache. Keep separate
cold replay tests for both public routes and QR/nonlinear recovery. New source
assembly is required before numerical pricing; source-19 remains immutable.

Skeptical audit passes this engineering repair: the cache is a bounded memo of
actual checked calculations, not persisted caller assertions; current bytes
are still hashed, binding identity includes runtime/source policy, and the
rung schedule is explicitly included because it is not part of execution hash.
The weakest assumption is that equal keys imply identical reconstruction
inputs, so negative controls must target all three components. The measured
551.481 seconds is an upper bound on potential removed work, not a forecast or
proof that confirmation becomes affordable. Other family prices and independent
confirmation remain required.

### Next complete family cases after frozen-source checks

The scoped replay repair passes 21 local checks, including both public routes,
exact standalone/scoped agreement and corruption/isolation controls. Another
100 existing grouped-replay, seed-validation, confirmation and pricing checks
pass. Source-20 is assembled; its focused and actual QR/nonlinear fresh-process
recovery checks must finish before numerical execution.

After those pass, run the two missing full-procedure development families,
nonlinear SSM and supplied residual-whitened funnel, on source-20. The question
is whether their original complete search and all-member closeout deliver;
these have not yet had a complete current-procedure price. Repeat neither
successful QR development merely for favorable timing nor failed confirmation
seeds. Keep original family base seeds `(20261002,2502)` and `(20261002,2503)`,
all model data/starts, six-L primary grid, L4/7 refinement, M100, W3/T65,
32--256 repetitions, epsilon hypotheses, vetoes and all-member retention.

Use the existing 1,800-second search cap and combined 3,000-second process cap
(1,800 search allowance plus 1,200 closeout allowance). Reserve at most 6,200
GPU seconds for the two sequential attempts including overhead, inside the
current release balance. The allowance is inherited from the complete QR
attempt as a development hypothesis, not a measured price for either family.
Timeout, empty delivery or partial closeout is an unsuccessful price and a
repair trigger; invalid evidence/source/device or unknown harness failure stops
the affected queue. CPU regressions cannot supply GPU eligibility; each worker
still verifies actual GPU/XLA/memory growth and all persisted chunks.

Command, conditional on both frozen suites passing:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_hmc_v7_release_prices.py --source docs/plans/artifacts/hmc-v7-release-2026-10-02/source-20/source --output docs/plans/artifacts/hmc-v7-release-2026-10-02/gpu-prices-scoped-closeout-01 --gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c --cases nonlinear funnel_residual --trial-batch-size 32 --wall-seconds 1800 --closeout-seconds 1200 --budget-seconds 6200
```

Skeptical review: the fixed family-to-seed mapping is already tested, all new
bindings reconstruct raw evidence at least once, and checkpoint/read integrity
is independent of the arithmetic memo. These two cases answer the remaining
functional question and supply missing cost evidence. They do not establish a
same-source three-family forecast without a matching QR price, or make the
32-per-family confirmation affordable. Retain that unfavorable forecast and
freeze no confirmation design yet. After these attempts, refresh the full cost
and gate table before deciding among further measured engineering repair,
a transparently revised replication allocation with the same statistical
criterion, or additional compute. Do not silently reduce a denominator or
spend the separately funded SSM reservation; its latest ledger remains running.

Pre-price code review found one further cache input: the current reader's
predecessor inventory. Equal final-record bytes must not stand in for a missing,
duplicated or changed predecessor in another reader. The shared-cache hit now
runs the existing predecessor identity/hash/analysis check before returning.
Three explicit negative controls cover those cases. No saved numerical result
was affected; this was found before source-20 pricing. Source-20 is retained as
an intermediate tested assembly; use a new source-21 assembly containing this
additional check for the two-family command above. The CPU phase ceiling remains
900 seconds, and no GPU price has started. All launch conditions still apply.

Source-21's 37 focused checks pass, including the added predecessor controls,
both public routes, windowed preparation, position-field mechanics and attempted
work accounting. The isolated source-manifest SHA-256 is
`e174198d1163725f2fb2b635a76ccb1bfde999e5019c84418da80acb844db851`.
The bounded CPU repair phase has used 696.679 seconds. Its source-20 two-model
recovery took 214.856 seconds, exceeding the 203.321 seconds left in the original
phase cap. Amend this convenience phase ceiling from 900 to **1,000 CPU worker
seconds**, within the unchanged 21,600-second release allocation, to allow one
260-second final source-21 recovery check and reconciliation. This measured
allowance avoids knowingly underfunding the required check. No scientific
criterion, campaign ceiling, GPU allowance or underlying recovery fixture changes.

Both final source-21 recovery cases pass in 196.642 CPU seconds. The complete
repair phase used 893.320 CPU seconds, below its amended 1,000-second ceiling.
The 37 focused and two recovery checks, including all predecessor controls,
satisfy the launch conditions. Reserve 6,200 GPU seconds for the exact two-family
command above with **source-21/source** substituted for source-20/source.
Both source snapshots and all prior receipts remain preserved. No confirmation
or default promotion is authorized by these development cases.

### Confirmation-report input audit during the two-family run

The running source-21 queue is unchanged. Read-through of the new diagnostic
reporter found that it checked the outcome wrapper's seed but did not compare
`model_result.sampling_streams` with the planned seed. An accidentally copied
model result could therefore masquerade as another independent replication.
It also checked only part of the declared full-search configuration. Before any
confirmation design is frozen, make the report require the actual model seed
and the complete shared full-search configuration, including all numerical
policy fields, epsilon hypotheses, refinement controls and resource settings.
Use the existing framework-free configuration constructor as the comparator;
keep the caller's declared cap explicit. The execution batch size, data and
geometry remain checked by the execution/configuration manifests; the model
result does not carry those complete identities.

Add negative controls for relabeled/repeated model seeds, missing seed identity,
changed error allocations, evidence counts, qualification bounds, epsilon/grid
settings and other search controls, plus an actual saved source-19 QR positive
control. No raw-evidence authentication is claimed: file checks and numerical
replay remain the readers' responsibility. The reporter only rejects internally
inconsistent inputs rather than treating wrapper labels as data.

This is diagnostic reporting only, no sampler/statistical-formula/default change.
Allow at most 60 CPU worker seconds from the existing reconciliation reservation.
Tests retain the exact rational binomial reference checks. Save artifacts under
`confirmation-input-audit-01`; preserve the running source-21 tree. Skeptical
audit passes: the defect concerns the independent-unit denominator and actual
method comparator, and is corrected before any claimed confirmation outcome.

The reporter repair passes 104 combined confirmation/pricing checks. The actual
source-19 QR result is accepted as a complete QR price, while a full forecast
remains unavailable because the other families are missing. Tests and that CLI
check cost 2.286 CPU seconds, charged once. The reporter snapshots are preserved
in their result roots; the running source-21 runtime and prices are unchanged.
No confirmation has been launched, and no reported development result has been
reclassified as an independent replication.

### Maintained regression selection audit, October 4

The maintained acceptance integration command omits the new scoped replay,
emitted-analysis reuse, trial-validation reuse and confirmation-report tests.
Individual receipts establish that those tests ran, but a future user of the
documented selection would miss their checks. Register the four files in that
selection and add their failure mechanisms to the coverage inventory. Keep the
longer replay tests in the integration tier; the confirmation reporter is
framework-free but belongs beside the existing pricing checks in this command.

This is a test-discovery repair, not a numerical or scientific-policy change.
The comparator is the actual file inventory and pytest collection, with no
inference from test counts to sampler reliability. Verify collection of the
maintained selection and run the inventory/reporting tests, capped at 180 CPU
worker seconds from the release balance. Preserve source-21 and its running
queue. No new long numerical run is required for registration alone.

Skeptical audit passes: the observed omission is concrete; the existing tests
provide behavioral and independent arithmetic oracles; collection exposes
missing imports/files before a long suite. A successful collection establishes
discoverability only. It cannot close full-procedure delivery, affordability,
independent replication or GPU gates. Fix the stale source-19 next-action field
in the release progress record so that it names the actual source-21 session.

### Confirmation feasibility audit before further cost repair

The source-21 nonlinear search reached its 1,800-second search deadline with
10 verified candidates, 11 promotion failures and nine unfinished candidates.
Its all-member closeout is still running. This is a partial search, even if
every exported member passes reconstruction. Preserve the failed development
price; completion or timing of the residual-whitened funnel remains unknown.

Before revising a replication count, quantify the exact design constraint.
For three families with equal Bonferroni allocations and the inherited strict
0.80 lower-bound requirement, even all-success evidence requires
`(4/5)^n < 1/60`. The smallest possible n is 19: n=18 fails. This is a necessary
optimistic design floor, not an adopted denominator or an adequate-power claim.
The current n=32 proposal permits one failure per family. Reducing n without
showing power and costs could produce a cheaper but unlikely-to-pass campaign.

Use the existing independently tested rational binomial helper to tabulate the
critical success count and pass probability for n=19,32,64,96,128 under true
delivery probabilities .90,.95,.99. These probabilities are sensitivity
assumptions, not measured success rates; the counts other than 32 are planning
alternatives, not new policies. Preserve both per-family power and the joint
power under independent family searches, stating that independence assumption.
Also show the optimistic average complete-search cost that fits the current
uncommitted GPU balance, before any further diagnostics or contingency reserve.
This arithmetic can rule out a proposed allocation at measured costs, but
cannot certify affordability from incomplete or different-source prices.

Save the deterministic calculation and source snapshot under
`confirmation-design-feasibility-01`, using the CPU receipt wrapper with a
30-second cap and no framework/device initialization or sampler. Skeptical
audit passes: it retains the original reliability criterion, does not choose
a denominator after confirmation outcomes, labels all unknown rates as
hypotheses and cannot authorize a run. Once the current queue settles, use its
actual cost attribution to select a bounded repair with a credible path to
the necessary cost range. Old-source profiles and small kernel timings alone
are insufficient reasons for another full price. A new complete current-source
forecast and frozen seed inventory remain required before confirmation.

The nonlinear worker has now settled with partial delivery: all 10 verified
members exported/reloaded, checkpoint reconstruction and attempted accounting
passed; 5,536 complete trials, no numerical invalidity or unaccounted attempted
work. Enclosing worker time was 2,208.007 seconds. The queue continues with the
original funnel seed. This cost remains inside the live 6,200-second reservation
until its enclosing terminal receipt is charged once.

Apply the earlier recorded-span diagnostic to this settled nonlinear result,
with explicit support for its partial-search classification. Check the 5,536
chunk digests, work/seed inventory and model accounting; sum native batch-call
shares exactly once and distinguish serialization spans from uninstrumented
search time. No extrapolation to complete search is allowed. Save source and
result under `source21-nonlinear-cost-accounting-01`, with GPU hidden and a
60 CPU-second cap. This is a read-only diagnostic of a settled case; it may run
while the independent funnel worker continues. The audit passes because the
baseline is now source-21's actual evidence, not the stale source-15 profiler,
and the script must retain `partial_budget` in the result.

### Exact work-metadata validation audit

Code inspection found a separate input-validity gap in `validate_chunks`:
it compares only fields supplied by a chunk's work record. Missing fields,
including candidate identity or the trial horizon, therefore escape that
comparison; an unexpected null field also compares equal to the null fallback.
The intended comparator is the complete repository-issued work payload,
ignoring only its mutable execution status. Other scope/seed checks remain
necessary but do not justify treating incomplete work metadata as complete.

First reproduce omission of each required field, an empty work mapping and an
unexpected null field using the existing codec fixture. Then compare complete
field-to-digest mappings, preserving tuple/list normalization and the intentional
status exception. Check unchanged valid records, changed status, and every
existing row/charge control. Run the focused validation/emission/replay and
trial integration selection with a 240 CPU-second ceiling. Preserve failing
pre-repair test output; no schema, seed, acceptance arithmetic or public API
changes. Source-21 remains frozen; its actual nonlinear chunk-work records
already passed complete equality against their issued records in the recorded-
span audit, so this finding does not invalidate that partial result.

Skeptical audit passes this localized scientific-record check: its baseline is
the complete issued work, its negative oracle is missing or changed identity,
and its positive controls retain supported status transitions and JSON list
normalization. The fix does not establish target correctness or statistical
reliability. Any later numerical launch must use a new checked source assembly;
do not modify or restart the running source-21 funnel.

### Source-21 closeout and the next discriminating cost check

The two-family queue settled at 3,589.238 GPU seconds. The original nonlinear
case is partial with 10 checked members; the original residual-whitened funnel
completed with 18 checked members out of 35 candidates and 9,600 valid trials.
Both reconstructed their checkpoints and accounted for every attempted chunk.
The release ledger is charged once and its 6,200-second reservation released.
The release has 29,820.817 GPU seconds and 1,906.029 CPU worker seconds remaining,
including the existing 250 CPU-second reconciliation allowance. The separate
old-policy SSM service reports closed; its records are not v7 evidence and no
refund is presumed here.

The nonlinear recorded spans attribute only 224.074 seconds to native batch
calls and 255.535 seconds to native calls plus tensor serialization. Its
1,750.990-second search leaves 1,495.455 seconds uninstrumented. That residual
cannot all be called a GPU bottleneck or cache cost. A trusted point observation
found two processes on the diagnostic GPU, so contention remains a possible
explanation, not a measured causal share. The source-15 profile is stale after
the subsequent seed, emission and scoped-replay repairs.

Use the existing saved-trial assembler profiler on source-21's first completed
nonlinear work item (smallest recorded ordinal, not a selected favorable case).
Reconstruct its original raw trials and require exact serialized equality;
call no target/HMC transition and grant no replay authority. Freeze the input
path and hashes and use the same verified GPU/XLA/memory policy. Cap the
enclosing diagnostic at 60 GPU seconds, save it under
`source21-nonlinear-assembly-profile-01`, and charge its enclosing receipt.
The existing profiler limits its worker to 55 seconds. This is a current-source
localization check; cProfile timings include instrumentation and cannot forecast
confirmation. If it fails exact parity, investigate before any runtime repair.
If reconstruction is cheap, inspect the controller/checkpoint host path rather
than repeating arithmetic batching already rejected for exact-parity failures.

This audit passes the smallest-useful-check criterion: actual settled nonlinear
evidence replaces speculative attribution; no old source is used as a current
timing baseline; the numerical output comparator is exact; and the diagnostic
cannot consume confirmation slots or silently modify its denominator.

The current-source saved-trial check passed exact equality for the first 32
nonlinear trials. It cost 9.462 enclosing GPU seconds; instrumented cold
assembly took 5.621 seconds, with 192 compiled-function calls. Compilation,
host dispatch and shared-device waits are included. This cold sample cannot
attribute all 1,495 seconds of residual search time or predict a full search.

### Complete source-21 evidence audit before the next source freeze

Run one framework-free settled-artifact audit, capped at 90 CPU worker seconds,
under `source21-settled-audit-01`. It must verify the frozen source manifest;
both original configurations/base seeds, batch size and GPU/memory policy;
the checkpoint's content/result digests; all chunk digests and exact issued
work fields except status; complete accounting and every verified member's
presence; and both terminal classifications. Preserve the partial nonlinear
outcome and all queue costs. File consistency and the saved successful numerical
reconstructions are distinct evidence: this audit does not re-run their kernels.

Then assemble source-22 once with the tested complete work-metadata check and
maintained test selections. The numerical work-metadata patch is three lines;
77 focused checks passed in 176.057 CPU seconds after the preserved failing
reproduction. No throughput optimization has yet been adopted. Budget at most
420 CPU worker seconds for assembly, collection and a bounded frozen-source
validation selection; do not re-run the long full integration manifest just
because its selection was repaired. This closes reproducibility of the patch,
not affordability. Keep the successful source-19/source-21 evidence immutable
and explicitly scoped. A source-22 numerical price remains conditional on a
reviewed cost-repair decision; do not buy another unchanged full price merely
because a new source tree exists.

Skeptical audit passes: no failed result is relabeled, every original denominator
and source is retained, and deterministic record checks cannot be mistaken for
fresh numerical replication. The remaining release gap is a complete affordable
current-procedure confirmation design. Its possible outcomes are a demonstrated
cost repair with bounded confirmation, or a documented funding/scope decision;
neither an engineering test count nor an additional successful single seed can
substitute for that decision.

While source-22 checks run, further subdivide the settled nonlinear residual
using its already-recorded `work_elapsed` events. Source inspection shows that
each event times outcome generation plus application, including inner chunk
checkpoints; the before/after work checkpoint writes are outside the event.
Sum these disjoint spans once, reject duplicate/unknown work IDs or nonfinite
times, and compare with the enclosing search and the existing per-chunk span
sum. The differences are explanatory residuals, not measurements of one
particular function. Save the framework-free arithmetic and input digests under
`source21-nonlinear-work-spans-01`, using a 30 CPU-second cap. This read-only
check may nominate host-checkpoint profiling if the outside-work residual is
large; it cannot justify weakening persistence or charging rules.

That partition attributes 1,718.556 seconds to outcome generation/application,
including 1,463.021 seconds beyond native calls/serialization; only 32.434
seconds lie outside work-item spans. This directs the next diagnostic inside
the work item. On the same first saved nonlinear work, time the existing
replicated-statistics and reporting-only R-hat routines separately, one cold
and two warm calls. Require exact JSON-normalized equality with the saved
analysis and R-hat, respectively. Use source-21, retain the original inputs,
and preserve the normal tensor placement instead of forcing a new device.
This reuses diagnostic inputs only; it does not issue evidence or rerun HMC.
Save under `source21-nonlinear-reports-profile-01`, at most 60 GPU seconds,
with verified memory growth and a 55-second enclosing subprocess deadline.
Timing differences remain descriptive and cannot replace the full-price gate.

The skeptical audit rejects assuming that R-hat is the cause merely because
it is expensive elsewhere. This targeted comparison either identifies its
actual cost here or leaves the cause unresolved. All report values and
acceptance decisions must be preserved; a mismatch rules out the diagnostic
as evidence for a transparent cost repair.

Source-22 passed 186 focused frozen-source checks, and its maintained
integration selection collects 346 tests. The assembly diff contains only the
complete work-metadata repair, registered tests/inventory, the earlier audited
reporter changes, and the freshly built native library; its C++ source is
unchanged. Run the existing QR independent scalar-reference and nonlinear
likelihood-composition checks in this tree with a 90 CPU-second cap, within
the 420-second assembly/validation phase allowance. This addresses the changed
build product without repeating full HMC prices or assuming binary identity.

The saved-summary timing probe passed exact equality for every call. Warm
replicated statistics took .021--.027 seconds and R-hat .022--.023 seconds;
cold calls include compilation. These descriptive timings do not support
R-hat as the main cause of the 1,463-second residual. The remaining useful
diagnostic is an instrumented work item on the actual public route, including
trial analysis, inner persistence and host serialization, with its raw output
compared to an uninstrumented same-seed execution. It must remain a bounded
development check and precede any new full-price campaign. Do not keep adding
whole-search repeats when the mechanism and achievable savings are unresolved.

### Bounded actual-work-item profile

Execute exactly the first issued work item of the original nonlinear full-grid
configuration through the public ordinary tuner, using its existing
`max_work_items=1` debugging limit. Run two fresh source-22 processes, one plain
and one with inclusive component timers and cProfile. Both use the original
base seed, frozen geometry/data and 32-trial first rung. The full search remains
declared; this intentional early stop is a diagnostic partial search and must
not count as a complete price, delivery success or confirmation slot.

Instrument native execution, trial validation/assembly, health, statistical
summaries, reporting-only R-hat, checkpoint writes and observation application.
Do not change their implementations or persist manufactured evidence. Compare
all original per-trial seeds, samples, traces, health, scores, full analyses and
candidate states with the uninstrumented process exactly; ignore only recorded
durations when comparing. This tests that the profiler preserved the numerical
work. Inclusive component spans overlap and must not be summed; profiling
overhead and cold compilation preclude a speed-ranking claim. One work item
does not price accumulated-history costs later in a search.

Reserve 400 GPU seconds inside the release balance, two subprocess ceilings of
180 seconds plus parent/closeout allowance, at
`source22-nonlinear-work-profile-01`. The cap is a convenience diagnostic
allowance informed by the preceding saved-record and reference costs, not a
measured full-work price. Use source-22's checked build, trusted GPU access and
verified growth before import/use; save commands, source/configuration hashes,
device settings, original seed and every subprocess outcome. Stop this
diagnostic for timeout, invalid device/source, missing evidence or exact-parity
failure. A passed profile may nominate one cost repair; its implementation still
needs focused validity/recovery checks before any new complete price.

Pre-execution audit passes: the baseline is the same actual source/model/seed
without instrumentation, the first issued item is chosen before its outcome,
no sampling operator or decision criterion changes, and the deliberate partial
scope is explicit. This does not resolve or relax the independent-confirmation
gate. Numerical invalidity invalidates this diagnostic; a slow valid first
work item remains evidence for repair rather than rejection of the method.

The two first-work processes passed exact equality of every trial seed,
sample, trace, health record, score, analysis and candidate state. Their total
charge is 255.311 GPU seconds. The instrumented `work_elapsed` span was 28.562
seconds, whereas the inner replicated `observe` span was 15.949 seconds and
observation application negligible. All eight checkpoint writes together
took .146 seconds in this first-work diagnostic. Source inspection locates a
potential missing 13-second component before the inner observer:
`HMCCandidateExecutionBinding.observe` calls `validate`, which calls `_probe`
on four scalar start-bank values/scores and target telemetry. This is a
candidate cause, not yet a measured attribution, and first-work persistence
costs still do not establish late-search costs.

Time exactly that unchanged binding validation three times on the saved
source-22 plain-worker execution spec: construct a fresh real binding, then
time its `validate` and the existing `_probe` subcall without replacing either
calculation. Source, geometry, capability, starts, device and returned probe
digests must remain equal to the stored spec. Use the actual target from the
saved model configuration, no samples or acceptance decisions. Save under
`source22-nonlinear-binding-profile-01`, with a 150 GPU-second cap and a 140-second
subprocess timeout. Preserve numerical source-22. This is the smallest next
check of the missing component; it does not authorize caching or skipping
target-mutation detection. The audit passes because a candidate cause is
measured directly on the same real binding and no mutable-state check is
removed or replaced by a caller assertion.

### Latest checked result and next repair hypothesis

The real-binding diagnostic passed all three unchanged validations, reproducing
the exact stored value/score/status probe each time. It measured 8.954--9.529
seconds in `_probe` within 8.966--9.538-second validations. Its enclosing charge
was 44.871 GPU seconds. This identifies a material repeated cost: every work
item calls four scalar start-bank value/score evaluations plus telemetry before
the inner trial observer. It does not prove the entire late-search residual or
a speedup for a replacement. All release diagnostics are now terminal.

The next repair hypothesis is to preserve this **current-target** probe with
less repeated execution. First inspect the batch capability of the repository
ordinary/fixed-transform adapters and compare a native four-start call with
the exact existing scalar-probe output on Gaussian, QR, nonlinear and supplied
residual-funnel fixtures. Native batched evaluation would continue calling the
current adapter on every validation, so it need not weaken mutation detection.
It may nevertheless differ numerically or fail on a supported scalar adapter;
those are real compatibility checks, not grounds to relax the comparator.
Keep the scalar path when batch capability is absent or not established.
No batch implementation or cached target-result shortcut has been adopted.

Before implementation, specify its exact capability rule and paired diagnostic
command; cap that diagnostic at 180 GPU seconds within the existing balance.
Exact value/score/status parity, unchanged source/geometry/target-mutation
detection and actual both-route regression coverage are prerequisites for
adoption. If this cannot preserve the contract, reject it and inspect a stable
graph with explicitly bound mutable inputs; never cache the stored answer or
skip the probe. A repair must then supply a complete downstream price before
any affordability claim. The master retains the full scientific procedure and
all three reliability families.

The first diagnostic now has a concrete capability boundary: the ordinary
affine wrapper may delegate a rank-2 probe only when its underlying target
explicitly declares `batch_rank_policy="rank2_required"`; the fixed-transport
wrapper additionally requires its existing native-batch transport support and
a declared batch target. Undeclared scalar targets keep their existing calls.
Gaussian and residual-funnel fixtures are diagnostic comparisons even when
they lack that declaration; successful sampled equality does not itself issue
a new general capability. The paired diagnostic uses source-22, each family's
original four starts and prepared geometry, and current telemetry. It records
both complete outputs and exact equality, including per-field differences.
No persistent target-result cache or mutation of frozen source is permitted.

Command: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python
docs/plans/artifacts/hmc-v7-release-2026-10-02/probe_batch_diagnostic.py
--source docs/plans/artifacts/hmc-v7-release-2026-10-02/source-22/source
--output docs/plans/artifacts/hmc-v7-release-2026-10-02/source22-probe-batch-01
--gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`.
The 180-second enclosing GPU reservation includes a 170-second worker timeout.
This tests the unchanged eager mutation probe, an explicit debugging exception
to the numerical-kernel XLA default; it does not change the HMC/XLA path.
Exact output mismatch rejects this replacement for that route; shape, finite,
source, device or allocator failure stops the diagnostic. Timing is explanatory
only. Skeptical review passes for this diagnostic: no selection, admission,
seed, horizon, grid, target or comparator changes, and the output can answer
the exact-parity question without a new sampling run. A pass still requires
live-mutation regression tests and complete downstream pricing.

The direct batch diagnostic finished in 27.495 GPU seconds. Gaussian and the
supplied funnel matched exactly; QR and nonlinear values/scores did not. The
replacement is rejected for the state-space routes. No runtime code changed.
The nonlinear fixture instead violates the existing recurrence helper's caller
contract: its eager likelihood calls construct a fresh compiled recurrence
four times per probe. The next bounded diagnostic wraps that same likelihood
in one stable graph, keeping singleton evaluation and the inner XLA recurrence.
It compares the full scalar probe exactly before and after, repeats the graph
probe, and passes changed observation tensors explicitly as inputs to test
that the graph does not retain an old answer. This is a fixture-specific
compilation investigation, not a generic compiled-adapter cache.

Run `probe_graph_diagnostic.py` under the same source-22/device environment,
output `source22-probe-graph-01`, with a 160 GPU-second reservation and 150-second
worker cap. The outer graph deliberately preserves the original XLA boundary;
it is a debugging comparison, with no sampling or public support claim. Exact
probe equality is still required. For adoption, all mutable numerical inputs
(including module model constants) must be passed explicitly, callable/source
drift must remain visible, and native chain output must remain unchanged.
Pre-execution review passes: the existing eager scalar implementation is the
comparator, numerical failure rejects the candidate, timing is explanatory,
and the diagnostic cannot establish complete-search cost or reliability.

The singleton-graph diagnostic passed exact full-probe equality, with one trace
and changed-data equality to the original eager calculation. Enclosing cost
was 24.190 GPU seconds. Warm probe times were .068--.082 seconds in this one
shared-device diagnostic, versus 13.032 seconds cold for the scalar baseline;
this is not a complete-search speed estimate.

Implement the repair in the Model B fixture likelihood: reuse a bounded graph
per static batch/horizon/backend, passing theta, observations, alpha and
observation sigma explicitly on every call. Keep the existing inner XLA
recurrence boundary and keep native HMC tracing the original implementation.
The outer graph is a reviewed eager-inspection exception to full XLA: its purpose
is exact legacy probe arithmetic with bounded tracing, while the actual HMC
and recurrence numerical kernels remain XLA. Cache executable graphs only,
never values or adapter answers. A 16-entry LRU is a convenience host-memory
bound, unrelated to scientific thresholds; eviction may retrace and affects
only cost. Unknown shapes or non-JIT reference calls retain their declared
reference behavior. No general adapter cache or batch-probe change is adopted.

Pre-implementation skeptical audit: preserve current callback dispatch and
source/geometry/start validation; module constants become explicit inputs to
avoid stale captured Python state. Tests must cover repeated graph calls with
changed theta, data and both constants; same-label prior/likelihood mutation;
both public routes; finite-difference SSM references; and fresh-process SSM
recovery. Native numerical streams must match the source-22 first-work artifact
on trusted GPU exactly. Freeze a new source only after focused tests. Reserve
up to 650 CPU seconds for assembly and these checks (additional budget requires
an explicit ledger amendment inside the existing grant); cap a fresh-source
GPU parity diagnostic at 250 seconds. Stop adoption for an equality, mutation,
source, memory, recovery or reference failure. Successful adoption triggers
a complete nonlinear price under the unchanged original search, followed by
source-consistent QR/funnel pricing and the unchanged confirmation-design review.

The GPU parity command is `saved_stream_probe_parity.py --source
.../source-23/source --output .../source23-saved-stream-parity-01 --gpu
GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`, relative to this plan's artifact root.
It replays all 32 original native trial seeds from source-22's first public work,
compares every raw sample and trace exactly, and compares the saved scalar probe.
This direct runner comparison issues no tuning authority and does not resume
an old source-bound checkpoint. Fresh public runs preserve the original base
seed; their per-trial seeds are correctly regenerated from the new source-bound
scope. Requiring equal freshly derived seeds across different source scopes
would contradict the current provenance contract, so the parity diagnostic
passes the saved seeds explicitly instead. This distinction was checked before
launch. The 250-second reservation includes the 240-second worker cap.

### Source-23 adoption and downstream cost check

All 41 frozen-source checks passed in 156.798 CPU seconds; assembly cost 53.269,
and the preceding 12 worktree checks cost 75.662. The saved-stream GPU check
passed in 19.573 seconds, reproducing the original scalar probe and all raw
samples/traces for 32 independent trials. Source-23 differs from source-22 only
in the Model B eager inspection graph, its focused tests/inventory/selection,
and the rebuilt native binary (unchanged C++). The bounded repair is adopted.

Run `scripts/run_hmc_v7_release_prices.py --source
docs/plans/artifacts/hmc-v7-release-2026-10-02/source-23/source --output
docs/plans/artifacts/hmc-v7-release-2026-10-02/gpu-price-stable-probe-01 --gpu
GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c --wall-seconds 1800
--closeout-seconds 1200 --budget-seconds 3100 --trial-batch-size 32 --cases nonlinear`.
Retain the original search cap, six-L/M100 settings, base seed, all starts,
data, measurement/verification ladders, repairs and refinement. Preserve the
failed source-21 price. A complete cost includes every verified export/reload,
fresh checkpoint reconstruction and attempted-work accounting. A partial
search stays unsuccessful, regardless of delivered members.

Skeptical audit passes for the full development price: exact original-stream
evidence supports the graph repair, no statistical settings changed, and
3,100 enclosing GPU seconds fit the balance. This is a descriptive new-source
price, not a paired speed claim (derived trial seeds include source identity),
independent confirmation or release. Stop for invalid source/device/memory,
missing accounting, process cap or exhausted reservation. Complete nonlinear
delivery triggers current-source QR/funnel prices; another valid capped search
triggers investigation of its recorded cost before repetition.

While that worker runs, reconcile the closed old-policy SSM service by reading
its terminal execution receipt and grant ledger. This is a bounded standard-library
accounting check (30 CPU-second cap) under the existing reconciliation allowance.
Require one matching enclosing receipt and agreement of grants, charges and
balance; preserve the existing separate 1,200-second allowance and 600/1,800
acceptance charges. Record only the unused part of the original SSM reservation.
Do not transfer funds or change v7 evidence status in this check. Audit passes:
no scientific result is reused, nested fits are not charged twice, and a closed
service receipt rather than process-list absence establishes settlement.

### Interrupted source-23 closeout and bounded recovery

The source-23 nonlinear search completed in 826.935 seconds, with 45 candidates
and 19 verified members. All 19 exports finished by 908.215 seconds. The agent
then incorrectly treated an unprivileged process-list absence as evidence of a
stalled worker and sent Ctrl-C to the launcher during fresh member replay.
Both parent and worker recorded KeyboardInterrupt. A later **trusted** process
check confirms both terminated. There is no complete original worker receipt;
the 3,100-second enclosing reservation is charged conservatively in full.
This is an agent interruption, not a timeout, numerical failure or failed
sampling search. It still does not count as a complete end-to-end price.

Recover only member/checkpoint replay and accounting on unchanged source-23,
using all original saved chunks. Output `source23-closeout-recovery-01`, reserve
1,500 GPU seconds with a 1,440-second worker cap. Use the same trusted GPU and
verified growth, source/configuration hashes and all 19 exported member paths.
Compare reconstructed states/inventories and full attempted accounting. No
sampling, new candidate decisions, seed changes or checkpoint mutations.
Retain inclusive timers for the two small immutable-summary validators to
localize residual host cost; timing is explanatory only. Full recovery plus
the conservatively charged interrupted attempt supplies a cost upper envelope,
not an uninterrupted price or confirmation slot. Any independent confirmation
must count both interruption and recovery cost under its own declared rule.

Audit passes: the saved search is terminal, all exports exist, and source/device
identity is unchanged. Recovery repairs infrastructure without retuning. Stop
for source/device/memory mismatch, corrupted evidence, failed numerical replay,
missing charge or the worker cap. Do not interrupt solely on an unprivileged
process listing again; rely on terminal receipts or trusted process state.
The separate SSM accounting audit passed after correcting a `phase`/`status`
field typo. It finds 1,665.699 seconds of unused reservation and 9,278.015 GPU
seconds outside the release allocation; no transfer or evidence reuse occurred.

Recovery passed: all 19 members reloaded, checkpoint states matched, and all
10,816 completed trials/accounting were checked. It cost 444.563 GPU seconds;
the original interrupted attempt plus recovery has a conservative charged cost
of 3,544.563 seconds. The successful search itself remains 826.935 seconds;
do not present the conservative combined charge as its uninterrupted runtime.
The two immutable-summary validators occupied 37.263 and 22.793 inclusive
seconds of recovery. That identifies a remaining component, not all overhead.

The next cost diagnostic should be bounded rather than another full search:
reuse 32 recorded complete nonlinear trials and compare existing host-side
evidence construction against grouped TensorFlow reductions/validation with
unchanged arithmetic. The earlier flattened-chain proposal changed normalized
return values and is rejected. Candidate repairs must preserve each trial's
original shape and reduction order, all reason labels, boundary comparisons
and serialized fields. Use the first complete saved work (preselected by
creation order), including invalid/corrupted/near-boundary test controls before
adoption. Cap initial saved-data profiling at 120 GPU seconds and 60 CPU seconds
within current balances; no new full prices or confirmation are launched by
this diagnostic allocation.

This refines the earlier immediate QR/funnel-pricing step: even the nonlinear
search-only component at 826.935 seconds makes 32 repetitions cost 26,461.927
seconds, exceeding the remaining release allocation of 25,886.445 seconds
before any closeout or other family. This is a descriptive budget warning, not
a universal lower bound (source-bound seeds change workload). Preserve all
families/grid/horizons/criteria and investigate the remaining overhead before
spending on further complete prices. The 9,278.015 seconds outside the release
allocation have not been transferred. Any final confirmation design still
requires complete current-source costs, explicit power and all denominators.

The first saved-data diagnostic compares sequential assembly with two/four
host workers over the **same** 32 independent recorded trials. Each worker
calls the unchanged `analyze_trial`, score reduction and serialization on its
original `[68,4,3]` tensors. The map preserves trial order; it neither batches
chains into a different reduction nor calls a target or sampler. This tests
whether host dispatch is serialized unnecessarily while preserving exact
numerics. The existing four-CPU affinity and framework thread bounds remain.
Use fresh source-23, one warmup plus three descriptive timing repeats, exact
full-record equality and saved source checks. Two/four are convenience
concurrency hypotheses bounded by the existing CPU allocation, not defaults.
The test has no tuning authority. Stop on mismatched output, exception,
source/device/growth failure or the 110-second worker cap (120 GPU seconds
reserved). Passing only nominates a concurrency repair; mutation, ordered
failure/persistence and real-public-route regressions would precede adoption.
Audit passes: trials are independent immutable inputs, no randomized operation
is run, current values are recomputed every time, and comparison is against
the unchanged sequential full-payload implementation rather than a proxy.

The host-dispatch diagnostic passed exact full-record equality on all 32 saved
trials. Three sequential repeats took 3.195--3.426 seconds, two workers
2.120--2.274, and four workers 1.261--1.503. These ordered timings lack randomized
order/independent workload replication and are descriptive nomination only.
Decoded inputs were correctly CPU tensors from the standard archive reader;
the process had verified GPU access, while this is saved-data analysis, not
native GPU sampling throughput. One field-name typo failed in 3.640 seconds;
the corrected run cost 32.966. Both are charged within the 120-second phase cap.

Implement optional `trial_analysis_workers` (integer 1--4, default 1) in the
exact v7 execution configuration. More than one requires replicated evidence.
Keep default payloads unchanged; serialize the non-default setting and reject
it on incompatible legacy configurations. Thread only independent, already
decoded complete-trial health/score/serialization work, retaining original
trial order. Target calls, native sampling, checkpoint writes, accounting,
candidate scheduling and membership remain serial. Executor shutdown must join
all started work before propagating a failure; no worker may outlive assembly.
No thread-local source/device policy or live trust cache is shared as authority.

Tests compare complete sequential/threaded records and changed/bad trial rows,
exercise ordered exceptions and partial-row recovery in new processes, and run
both public export/reload routes. Update the model codec, pricing CLI, maintained
inventory and official/agent guide consistently; default execution stays one
analysis worker. Use at most 650 CPU seconds for implementation checks and
source assembly inside the current grant, then a 180 GPU-second exact saved
full-assembly parity check. Reprice only after those checks pass. Skeptical
review accepts this bounded optional repair: immutable per-trial inputs and
ordered completion preserve the numerical procedure; exception/lifetime and
device-context tests are material requirements, not assumed from the favorable
timings. No full-search affordability or reliability claim is made yet.

Final balance for this checkpoint: 1,637.480 CPU worker seconds and 29,502.265 GPU
seconds inside the release allocation, with 250 CPU seconds reserved for
reconciliation. Source-22 is checked; confirmation remains unfrozen. The plan
passes review as a bounded repair program, **not** as an affordable confirmation
plan. Release remains pending on complete current-source delivery, feasible
independent replication and the terminal support audit.

### Host-concurrency implementation audit, October 4

The resumed review found one material omission before testing: TensorFlow device
scopes and placement policies are thread-local. The worker now inherits both
from the caller; a narrowly contained eager-context read supplies the device
name because TensorFlow has no public getter. The first regression exposed a TensorFlow setter limitation: its global
cached value can suppress a required new-thread update. The revised worker
uses TensorFlow's native thread-local placement setter without mutating that
global cache. This small internal-API dependency has explicit regression
coverage and needs checking when TensorFlow is upgraded. Tests exercise explicit CPU placement even when GPU support is available;
trusted GPU parity must also cover explicit GPU scope. Default single-worker
execution does not enter this code. No result is cached. The test comparator is
the original sequential full-record assembly with identical raw tensors/seeds,
including invalid discarded prefixes, target-status failures, malformed evidence
and threshold-neighbour scores. Exception tests require input-order propagation
and completed executor shutdown. The revised bounded plan passes this review.

The previous paragraph labeled “Final balance” refers to the earlier source-22
checkpoint; the live allocation is authoritative. Recovery is terminal and
passed. Current balances are 1,342.730 CPU and 25,849.839 GPU seconds.

Source-24 manifest `2994fdc5ca7aa2cbac1a9a4611e361b7f0eb7ddc183535289642191efecf2a17`
contains only the reviewed host-analysis repair, model/CLI/inventory/tests and
a rebuild of unchanged native sources relative to source-23. All 82 frozen
checks passed in 173.693 CPU seconds, including both public routes retaining
two verified members and four QR/nonlinear fresh-process recovery cases with
four analysis workers. The complete-record GPU parity script is
`artifacts/hmc-v7-release-2026-10-02/trial_assembly_parity.py`. It reuses the first
issued source-23 work's 32 complete records and original seeds only as a
saved-data analysis fixture, not a new-source checkpoint or tuning artifact.
CPU output must equal archived records exactly; explicit GPU sequential and
threaded outputs must equal each other. Device-specific results are reported
separately. No new sampling runs. All record fields, including reason labels,
are compared; timing is descriptive only. Verified growth and source identity
are required. The reviewed 170-second worker cap/180-second reservation applies.

The first source-24 GPU assembly check failed its **comparator assumption**,
before threaded comparison: forcing CPU placement changed the old saved record.
The archived tensors decode on CPU, but that does not imply all original
operations executed there. Preserve the failed receipt (5.331 GPU seconds).
The revised check first reproduces the original default caller scope and must
match the complete archived records exactly. It then compares sequential and
four-worker assembly separately under explicit CPU and GPU scopes; cross-device
differences are recorded, not required to vanish. This restores the actual
original comparator rather than changing the sampler or acceptance criterion.
The revision passes skeptical review: equal placement is necessary for a claim
that concurrency alone preserves arithmetic. No new source or numerical change
is required. Retry stays inside the original 180-second parity allocation.

The second GPU check reproduced the original sequential records exactly, then
exposed an actual optional-path bug: `tf.device("")` resolved to CPU in a new
worker, although the caller's empty scope means automatic placement. This is
not a sampler or baseline failure. The new CPU regression failed before the
repair, showing empty caller scope versus explicit CPU in all four workers.
The repair uses no device context when the original name is empty; explicit
CPU/GPU scopes are still forwarded. Preserve source-24 and both GPU failures
(5.331 and 9.925 seconds); no source-24 price or tuning artifact was issued.
Source-25 must pass the regression, frozen public/recovery checks and the same
GPU parity before pricing. The remaining GPU retry cap is reduced to 150 seconds
so cumulative spending stays within the original 180-second phase budget.

The source-25 GPU parity runs four-worker assembly first, before the sequential
comparison, to test a cold concurrent first trace as well as steady replay.
Both still must reproduce the archived records under original placement.

### Full nonlinear price after source-25 parity

All 83 frozen-source checks pass (177.301 CPU seconds). Conditional on exact
GPU assembly parity, price source-25's original nonlinear full procedure with
base seed `(20261002,2502)`, batch32 and four analysis workers. Preserve all
six primary L values, M100, starts/data, W3/T65, 32--256 repetitions, four
repairs per family and original refinement. Use the existing 1,800-second
search cap and 1,200-second closeout allowance; reserve 3,100 GPU seconds
including parent overhead in fresh `gpu-price-trial-workers-01`. Required
output is complete search, every verified member exported and reloaded,
reconstructed checkpoint and attempted-work accounting. A partial result
does not provide a complete price. Timings are descriptive because source
identity and derived streams differ; no speed ranking is claimed. Source,
GPU/XLA, memory growth and numerical validity are mandatory. The same
continuation vetoes apply; preserve every failure and charge. This is the
smallest full-procedure price following the checked cost repair, not a
confirmation launch. Review passes without changing any release criterion.

Source-25 GPU parity passed in 11.919 seconds: all 32 full records match the
archive in original automatic placement, and sequential/four-worker records
match separately on CPU and GPU. Forced-CPU records differ from the archive
in their numerical diagnostics, as recorded; placement must stay in scope.
The final public-route audit found the optional host count was also accepted
by the mechanics-only position-field route, outside this repair's tested
exact-score scope. Reject that new non-default setting before preparation,
mirroring the existing exact-score batching/reuse guards, with a public-dispatch
regression. This does not change any existing default or position-field evidence
contract. Freeze source-26, run the focused guard/placement/codec tests and
repeat GPU saved-record parity; source-25 exact-route/recovery evidence remains
compatible because those paths are unchanged. Include an alternating warm
serial/four-worker timing diagnostic; one cold timing cannot nominate the
cheaper setting. These local checks remain inside the existing repair budgets.
Only then activate the full-price contract above using source-26.

Source-26 manifest is
`39a2f4bb4836210ececfb97c1f74e73d4129e701d542d5f9316889f3d4329ba9`.
All 79 focused frozen checks pass (12.202 CPU seconds), including the new
public rejection and existing single-worker mechanics checkpoint recovery.
Only that guard, its test and a rebuild of unchanged C++ differ from source-25,
whose 83 checks covered both exact public export/reload routes and four fresh
SSM recovery cases. Source-26 trusted GPU complete-record parity passed in
14.715 seconds. Warm alternating pairs took .581--.736 seconds sequentially
and .405--.504 with four workers. These descriptive timings nominate the
reviewed four-worker full price; they do not prove a general runtime gain.
All cold/default/CPU/GPU record comparisons passed under their stated scopes.

The 650-CPU/180-GPU repair reservations are closed at actual enclosing cost;
unused amounts are released. The full nonlinear price now has its 3,100-second
reservation. Source-24's failures remain preserved. Next obtain the original
QR/funnel prices on this same source only after full nonlinear closeout passes,
then recompute affordability and power without shrinking the confirmation
denominator.

The failed source-26 full-price attempt is charged as a harness receipt: 62.341
GPU seconds, with no complete price and no release evidence. The traceback
identified concurrent TensorFlow signed-proxy validation, not target arithmetic,
resource loss or acceptance criteria. `_TF_TRIAL_ANALYSIS_LOCK` now serializes
concat, health/score TensorFlow calls and tensor serialization while preserving
ordered worker execution and full join semantics. This is a correctness guard,
so the optional worker count is no longer presented as parallel TensorFlow
throughput. A focused GPU smoke must pass before reattempting the full price;
all source-26 evidence remains immutable.

A four-trial GPU smoke is required before another full search. It uses source-27,
verified memory growth and explicit `/GPU:0`, constructs the same health/score
path with XLA enabled, compares sequential and four-worker complete records and
performs no sampling. It is an engineering gate for the lock repair; success
does not price the full procedure or establish a release.

### Withdrawal of optional host concurrency after native-search failure

The source-26 failure is an observed TensorFlow reduction error during concurrent
analysis of work 17 (64 new trials, ordinals 64--127). The earlier description
that a TensorFlow race had been established was too strong: the underlying
cause has not been isolated. The source-27 lock is **not adopted**. It serializes
almost the whole task, defeats the proposed benefit, retains private TensorFlow
context dependencies, and conflicts with the ordered-failure barrier test. The
subsequent four-trial smoke failed in its own helper (`dataclasses.replace` on
a `SimpleNamespace`) before numerical comparison. It supplies no repair evidence.

The smallest justified repair is to withdraw this unneeded new public option,
restore the checked source-23 serial implementation and its original model/CLI
codecs, remove the advertised worker option from both guides, and preserve all
failed sources and costs. Inspect byte differences before restoration to avoid
unrelated changes. Add a compatibility check that experimental worker-bearing
configurations cannot silently reload as serial. All model, grid, horizon,
acceptance, error allocation and retention criteria remain unchanged.

First run an at-most-90-second saved-data GPU diagnostic on the exact failed
work, with the source-23 serial helper: reconstruct every original trial without
resuming or issuing an artifact, verify raw tensor checksums, compare repeated
serial output, and preserve complete health decisions. It is diagnostic only.
If serial analysis fails, stop pricing and isolate that failure. If it passes,
repeat the original full nonlinear price on the already checked source-23, then
price QR/funnel on that same source. Failed concurrency artifacts cannot enter
confirmation. The fresh price remains capped by the prior 1,800/1,200-second
search/closeout allowances and 3,100-second enclosing reservation. This decision
passes skeptical review because it removes a failed optimization without
changing the research target, its denominator or any promotion criterion.

The live GPU smoke command lacked an enclosing receipt and failed immediately
after framework initialization; conservatively charge 60 GPU seconds as an
accounting upper allowance, not a measured runtime. The long preceding tool
approval delay is not worker execution time. Subsequent diagnostics use the
existing parent wrapper with a monotonic enclosing receipt and timeout.

The exact failed work passed serial reconstruction: all 64 trials have valid
evidence, and three complete records agree exactly (11.306 enclosing GPU
seconds). This supports proceeding on the established serial implementation;
it does not isolate the TensorFlow error or validate a concurrent path. The
71-file runtime/driver audit is byte-exact against source-23. 57 compatibility,
inventory and documentation checks pass (6.735 CPU seconds).

Launch `gpu-price-serial-restored-01` using source-23's frozen price driver,
original nonlinear seed `(20261002,2502)`, original full grid/repairs/refinement,
batch32, serial analysis, search cap1,800 and closeout1,200 seconds. Reserve3,100
GPU seconds. It repeats the earlier interrupted development seed for complete
cost measurement, not independent confirmation. Preserve both earlier attempts
and all charges. No failed concurrent artifact is loaded or used as tuning data.

### Confirmation execution preparation while serial pricing runs

The existing exact-binomial reporter is not a campaign executor. Prepare a
small serial GPU supervisor that reads one frozen design, uses the same
`run_model` public procedure from the frozen source, records every planned slot,
and reports with `delivery_report`. Do not launch confirmation until all three
complete same-source prices and a reviewed affordable design exist.

The executor must validate32 slots per family, unique disjoint confirmation
seeds, unchanged family profiles relative to their complete development prices,
source identity and explicit total/search/closeout budgets. It must preserve
all96 slots even after resource exhaustion or a harness failure; those slots
remain unsuccessful. Unknown execution errors invalidate the campaign and
prevent a delivery verdict; they are not relabeled as numerical rejections.
Both worker and parent preserve manifests/commands, memory-growth verification,
actual GPU/XLA chunks, source checks, per-attempt and enclosing times. Use fresh
output directories and the existing monotonic process-timeout pattern. This
supervisor grants neither release nor default-promotion authority.

Tests use a fake child and clock to check all denominators, budget exhaustion,
harness stops, identity changes, profile drift and seed duplication without
launching numerical work. Reserve at most60 CPU seconds for these engineering
checks from the current release balance. Reuse100 seconds of parent settlement
allowance from the existing price-launch pattern as an explicit convenience
allowance in a future campaign design; it is not a measured runtime bound.
The review passes: there is no new sampler, success proxy, optional stop on
favorable outcomes or budget extension; final confirmation still depends on
affordable complete prices and frozen scientific inputs.

### Confirmation supervisor audit and affordability stop, October 4

The serial failed-work reconstruction passed all 64 records on three repetitions.
The uninterrupted source-23 nonlinear search has returned and exported all 19
members; its reload, checkpoint and accounting closeout must finish before it
is a complete price. The two supervisor test receipts passed (103 checks across
the reporter/supervisor, then 21 supervisor checks); these are engineering
checks, with no confirmation sampling.

Skeptical review identified two remaining supervisor defects. A corrupt result
read after child execution could prevent the enclosing terminal report. Also,
an immediate memory-contention deferral supplied no opportunity to recover
before numerical work. Repair the report to preserve every slot and classify
result/source/report failures as harness failures. Add bounded memory-readiness
polling before TensorFlow import; the frozen design must declare its wait cap,
poll interval and minimum free memory. Preserve every readiness observation,
and include waiting in slot and campaign charges. Do not retry a completed
numerical failure, replace a seed, or erase an interrupted numerical attempt.
After numerical work starts, a timeout remains an unsuccessful outcome; an
unclassified exception invalidates the execution report.

These resource controls are operational hypotheses, not admission thresholds.
The existing 4096-MiB price guard is inherited and must be explicitly chosen
with provenance in a future design; polling and wait limits have no new default.
Tests use fake inventories and clocks to expose memory recovery, persistent
contention, probe errors and timeouts. They cannot prove shared-device runtime
reliability. Use the existing 60-second CPU reservation for this focused repair.

Before spending on more prices, apply an early affordability check. If 32 times
one **complete current-source** family's measured cost already exceeds all
remaining authorized GPU time, the additive point forecast cannot fit even
if the other families cost zero. This is a descriptive funding screen, not a
lower confidence bound on runtime or proof that a future seed cannot be cheaper.
Report that precise gap and preserve resources for a justified repair or funded
design; do not run two more prices merely to establish the same shortfall.
Changing the 32-slot proposal or supported scientific scope needs a separate
explicit justification; this screen changes neither the denominator nor the
simultaneous delivery criterion. Keep independent engineering/release checks
moving when the confirmation funding gate remains open.

The audit passes for these local supervisor repairs and the early funding
screen: the numerical baseline, data, source, starts, seeds and criteria are
unchanged; corrupted evidence blocks interpretation, temporary resource
unavailability can recover before sampling, and no successful launch or proxy
metric is promoted to release evidence.

Terminal price audit: use at most 120 CPU worker seconds from the existing
250-second reconciliation reservation to check the completed source-23 nonlinear
receipt, raw chunk hashes and full work identities, device/memory policy, model
configuration, member inventory and source manifest. Compute the additive
funding screen from its enclosing charge and the reconciled ledger. This is a
read-only evidence/accounting audit; it does not resample or replace the fresh
numerical reload already performed by the worker. Write the checked inventory,
stage timings and funding calculation under `serial-price-audit-01/`.

The price and audit passed: 45 candidates, 25 repairs, all 19 verified members
exported/reloaded, complete checkpoint reconstruction, and all 10,816 native
trial chunks checked. Enclosing cost is 2,227.336864 GPU seconds. A 32-search
nonlinear-only point forecast is 71,274.780 seconds, while the entire reconciled
remaining GPU allowance, including untransferred funds, is 32,724.981 seconds.
The resulting 38,549.799-second shortfall excludes both other families and any
waiting/repair margin. This is a funding screen from one development seed, not
a measured full confirmation budget or a lower confidence bound on runtime.
Current-source QR/funnel prices and independent confirmation are therefore
deferred pending a justified cost repair or funded design. The implementation
and target were not invalidated; the independent delivery claim remains open.

Close the independent posterior-interface check while retaining that funding
veto. Reserve 400 CPU worker seconds from the existing release allocation for
the **original three K0 panels and seeds** plus the endpoint health controls,
on frozen source-23 with GPUs intentionally hidden. Use the maintained test
`tests/test_hmc_acceptance_posterior_integration.py` and
`tests/test_hmc_acceptance_endpoint_return.py`; preserve pytest's temporary
reference JSONs below the run directory. The pass condition remains sequential
warmup exclusion, independent retained streams, declared posterior assessment
and both location/variance errors within the existing four estimated MCSE
diagnostic tolerance. The tolerance is inherited reference-test policy, not a
coverage theorem. No panel, seed, budget or threshold changes are allowed on
failure. This closes compatibility on the current source, not SBC, universal
burn-in, or missing GPU confirmation. Audit passes because it checks an
independent open gate and cannot turn the funding veto into a release pass.

The current-source posterior/endpoint suite passed all 12 checks in 130.680 CPU
worker seconds. Preserve its three reference JSONs as diagnostic evidence, with
the original data panels and streams. Use at most 200 further CPU seconds from
the same 400-second reservation for the remaining scoped terminal checks on
source-23: the actual v7 windowed-preparation integration, the mixture
missing-mode negative control, adversarial acceptance/R-hat controls and the
official-guide/registry contracts. This is newly required current-source
coverage after the execution repairs, not a broader posterior-validity claim.
Failure preserves the source and original fixtures for diagnosis; a pass closes
these engineering/interface checks only. The funding and independent-delivery
gates remain open regardless of their outcome.

The terminal suite stopped during collection: source-23 omits
`scripts/render_hmc_tuning_interface_docs.py`, which the documentation tests
import. This is a validation-assembly defect, not a numerical failure; its
5.230-second failed attempt remains charged. Run the numerical/interface checks
on immutable source-23 without the missing documentation test, and repair the
assembler's explicit documentation inputs (renderer, inventory, examples,
generated tables, guide and official chapter). Include the confirmation
supervisor used by the maintained selection. Test a fresh **documentation-only**
assembly without rebuilding or sampling; verify its relevant runtime files
equal source-23 and record its separate manifest. This tree supplies guide/
assembly evidence only and cannot be mixed into price-source identity. Budget:
at most 100 CPU seconds from the existing reconciliation reservation. No native
source, guide prose, statistics, run configuration or saved evidence changes.

The fresh documentation assembly passes all 161 guide, supervisor, reporting
and inventory checks; all 71 relevant runtime/driver files equal source-23.
The current-source preparation, missed-mode and adversarial suite separately
passes 28 checks. A subsequent attempt to collect the entire native integration
selection in the documentation-only tree failed because it deliberately lacks
the compiled Sylvester library. That command was outside the tree's declared
role, not evidence that the documented `--build-custom-op` path is broken.
Preserve and charge its 8.186 seconds; check updated selection discovery in the
existing built workspace instead. Do not rebuild or mutate the measured source
merely to validate collection. Guide prose has not changed, so the earlier book
build remains scoped evidence; the new guide/registry contracts establish the
current agreement without claiming a new book rendering.

### Remaining-cost investigation after the complete serial price

The preceding goal turn made concrete progress: it completed the uninterrupted
nonlinear price, repaired the confirmation supervisor/validation assembly, and
closed 201 scoped checks. This continuation inspects the remaining cost problem
before another GPU launch. The baseline is the immutable source-23 complete
nonlinear price, not an older partial run or favorable microbenchmark.

Question: does the complete-price record identify a previously untested,
localized cost reduction large enough to justify spending on a repair before
confirmation? Read its recorded native batches, work stages and enclosing
timings. Verify each batch's row/seed inventory and count its native time once.
Separate first use of a runner shape from later calls; these are **recorded call
times**, not isolated compilation or uncontended device times. First-call cost
cannot be labeled compilation without further evidence. Preserve the target,
data, source, starts, grid, trial count, all-member checks and release criterion.

The research intent is cost localization, with no new sampler or numerical
decision. Diagnostics are explanatory/repair-nominating only. Source or saved
inventory inconsistency invalidates the analysis; a high valid cost is a
funding/repair finding, not a rejection of HMC. Compute the remaining-budget
capacity for `32 * (QR + nonlinear + funnel)` and explicit zero-cost-component
counterfactuals. Those counterfactuals illustrate the size of a required repair;
they are not runtime lower bounds, forecasts of an implementation, or permission
to skip work. A repair is nominated only with a concrete code path and a
discriminating next check. Existing tiny summary-speed differences do not by
themselves justify another full price; withdrawn concurrency remains excluded.

Allocate at most 40 CPU worker seconds, a convenience diagnostic cap within
the 531.515-second remaining release CPU allowance. Use the existing receipt
wrapper, GPUs deliberately hidden, and fresh root `serial-cost-components-01`.
No GPU time is reserved. The skeptical audit passes: this reads a complete
same-source baseline, counts disjoint spans without nested double charging,
preserves failed alternatives, and cannot promote timing or test counts into
release evidence. If no defensible repair is identified, retain the funding
blocker rather than launch an unsupported optimization or shrink confirmation.

The diagnostic passed in 3.423 CPU worker seconds. All 10,816 saved rows form
338 complete native batches, with no duplicate row or seed and one enclosing
native charge per batch. Its disjoint recorded partition is:

| Component | Seconds |
| --- | ---: |
| Native sampling calls | 628.468 |
| Native tensor serialization | 63.842 |
| Other work inside the search | 1,002.909 |
| Setup and all-member/checkpoint/accounting closeout | 532.118 |
| Full enclosing price | 2,227.337 |

The first native call for the single `(68 transitions,32 trials)` runner shape
took 9.323 seconds; the other 337 calls total 619.144 seconds. These observations
do not isolate compilation or resource contention. They do rule out attributing
most recorded native time to that first call. With 32 searches per family, the
remaining allowance permits a sum of three family prices of at most 1,022.656
seconds **before** readiness/settlement/development margins. The measured
nonlinear price alone is 2,227.337 seconds.

Even assigning zero time to any one of the four components leaves the
nonlinear-only point extrapolation above the whole remaining allowance. Removing
all uninstrumented in-search work, the largest category, would still leave
39,181.689 seconds for 32 nonlinear searches, versus 32,724.981 seconds available.
These are arithmetic counterfactuals, not permitted changes or lower bounds on
a different implementation. They show why another small host-summary optimization
does not by itself supply a defensible confirmation budget.

The inspected exact grouped-reduction diagnostic supplied only modest saved-data
differences and no full-record/native-run repair; the faster summary-batch label
in an earlier result is explicitly retracted in its `audit-disposition.json`
because its comparator/tolerance changed. Host threading is withdrawn after
native failure. None is eligible to fund confirmation. No new runtime repair
is nominated by this audit. A substantial combined throughput improvement remains
a possible research direction, with no checked implementation or measured total
price; it cannot be budgeted as if it already existed.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve current serial implementation and complete price | Saved batch inventory and disjoint costs checked | No new source or numerical defect found | Shared-resource contribution to wall time is not isolated | Keep the exact evidence and all failed alternatives | No arbitrary-device runtime guarantee |
| Defer further GPU spending on this confirmation design | No tested localized repair closes the affordability screen | Funding/affordability remains an open release gate | Complete QR/funnel costs and confirmation seed variation remain unknown | Require a concrete validated combined cost repair or changed funding direction before launch | No claim that every possible implementation is unaffordable |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Existing concurrency failure remains a veto on that optimization; no new invalid numerical trial |
| Statistically supported ranking | None |
| Descriptive-only differences | Saved component timings, first/subsequent calls and zero-cost counterfactuals |
| Default readiness | Unchanged; no release or default promotion |
| Next evidence needed | An affordable complete three-family design, independent fixed-denominator confirmation and terminal support decision |

Post-run review: simultaneous reductions in native and host costs, including
lower contention, could overturn the current funding screen. No existing result
quantifies such a reduction sufficiently to launch confirmation. The present
result concerns feasibility of the planned campaign under the remaining
allowance; it rejects neither the serial implementation nor the research target.

October 4 blocked-state revalidation: the source-23 price receipt and all 71
current runtime/driver files still match their checked evidence. All release
workers have terminal receipts and there are no active reservations. Remaining
GPU authorization is still 32,724.981 seconds, while the nonlinear-only
32-search point forecast remains 71,274.780 seconds. The earlier cost audit
identified no tested localized repair sufficient to close that screen. The same
condition has persisted through three consecutive goal turns. Mark the goal
blocked, not complete; retain all open P4/P5 work and the original objective.
No further blind price, untested optimization, reduced denominator or release
claim is justified by the existing evidence. Resumption requires a funded
scientific design or a concrete validated combined cost-repair direction.
