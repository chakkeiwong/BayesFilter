# q20 validation, staged calibration and budget repair

Date: 2026-09-16. Status: implemented and verified; fresh GPU timing unavailable.

## Question and evidence contract

Repair repeated validation work and premature whole-search reservations while
preserving the q20/T30 strict float64 target, reverse-KL objective, TensorFlow
batch-native GPU/XLA training, training seed streams and posterior checks.
The comparator is the current September 16 main working-tree implementation,
including the four-chain batched HMC repair. The old training cost calculator
reserves three maps times every validation bank size at every training rung.

Engineering acceptance requires equal cached/uncached losses on identical rows,
no evaluations of an already cached prefix, no cross-map/beta/target reuse,
safe exact optimizer continuation after a calibration stage or validation
interruption, and cost counts that match the executed schedule. A corrupted
cache, invalid target row, stale source, lost optimizer state, changed paired
bank or early map promotion vetoes the repair. Numerical failures remain
recorded; rows are never resampled to avoid failures.

The research question is how much learning and downstream sampling are needed
for this target. This repair does not answer it by inventing new iteration
counts. Calibration measures learning, loss-difference variance and cost;
these are explanatory diagnostics and repair triggers, not model ranking or
posterior promotion criteria. The existing full seed/architecture/LR inventory,
plateau/export policy, numerical validity screens, posterior convergence,
precision and reference comparisons remain required for their declared claims.
Resource exhaustion stops computation with a resumable checkpoint. A slow or
inconclusive training candidate does not reject the research direction.

## Implementation

1. Cache heldout loss prefixes for immutable map checkpoints. Identity includes
   map, target/bridge, beta, bank seed, batch/backend and source scope. Restore
   a frozen checkpoint for evaluation; never cache a mutable optimizer map.
   Extend only the missing suffix with the same stateless batch-index seeds.
   Persist checked TensorFlow tensors and ordinary checksums in the run output;
   copy the cache into a fresh output directory on resume. Check the cooperative
   budget between batches and retain completed prefixes on interruption.
2. Match validation work to its decision. Before the existing cohort floor,
   evaluate only the first existing bank: learning is still required and no
   early plateau qualifies a map. At/after the floor, stop on clear continuing
   improvement, clear deterioration, or a resolved fine-precision/plateau
   decision. Expand an ambiguous comparison through the existing bank ladder.
   At the update cap, resolve learning-versus-unresolved status without demanding
   irrelevant fine precision, but retain the fine plateau requirement. Preserve
   all intervals, decision reasons, evaluated/cached row counts and approximate
   rows needed for the requested precision. Reused normal intervals remain
   descriptive, not anytime-valid confidence claims.
3. Add an explicit `calibrate` master mode. It prices the actual training batch
   across every width/positive beta, then trains the first declared root for
   every width/LR in both schedules to the first existing rung. Direct training
   starts at beta one; continuation starts at the first positive beta. No
   continuation map jumps to the next beta in this stage. Its partial cohort
   resumes into the full unchanged protocol with exact seeds and Adam state.
   The stage cannot issue a development-eligible export, finish the cohort or
   bypass full-cohort and downstream checks. It does not require HMC pricing
   or HMC qualification because it performs no HMC. Full campaign execution
   still requires those measurements and qualifications.
4. Replace the misleading all-rung minimum with separate costs: calibration,
   optimizer floor plus first-bank validation, optimizer floor plus worst
   validation expansion, and complete update/validation caps. With R assessed
   maps and a baseline, caching requires at most R+1 distinct map banks per
   scope, not three maps times every nested bank at every rung. Price startup
   and compilation once per actual graph; avoid counting the first update
   again as a steady update. Keep the inherited factor two visibly identified
   as an engineering reserve, not measured runtime or a necessary lower bound.
   Missing downstream prices remain missing, never zero or a claimed complete
   campaign estimate. Calibration admission uses its own complete price, not
   a reservation for the entire eventual campaign.

## Default and assumption audit

| Choice | Provenance and purpose | Failure mode and early check | Status |
| --- | --- | --- | --- |
| Target, batch 32, widths 16/32, LRs .0005/.001, roots 0/1/2 | Existing protocol; hold mathematical and comparison scope fixed | Training may be slow/poor; measure actual q20 costs and learning before extending | Inherited hypotheses |
| Rungs 128/512/2048/8192 and floor 512 | Existing learning ladder; no convergence theorem | May stop too early/late; calibration records learning and MC uncertainty, full criteria remain | Uncalibrated hypotheses |
| Banks 768/3072/12288, delta .04, half-width .02 | Existing declared assessment policy | Overprecise or inadequate for downstream needs; report variance and implied row demand | Uncalibrated hypotheses, unchanged |
| First declared root and first rung in calibration | Deterministic execution staging of existing inventory | One seed cannot rank or prove robustness; no selection or promotion, full replication on resume | Engineering staging choice |
| First bank before floor | No export/plateau decision is allowed yet; avoid buying fine precision for a forced continuation | Missed deterioration if uncertain; preserve uncertainty and inspect before extending | Decision-specific diagnostic rule |
| Cache identity and immutable snapshots | Same deterministic finite computation on same rows | Mutation, beta/seed drift or corrupt tensor; explicit isolation/corruption/replay tests | Derived correctness requirement |
| Safety factor 2 | Existing reservation policy | Not a statistical runtime bound; report raw cost separately | Engineering hypothesis |

## Thorough preimplementation review

The initial idea of reducing the Cartesian grid was rejected: a short or
single-seed loss ranking would silently change the robustness claim. The
calibration stage instead changes execution order only. Direct and continuation
work at different temperatures is not compared as equal-target performance.
No smoke-trained map becomes a production mechanism.

Cache correctness requires more than candidate identity: the map changes at
each rung and the target changes with beta. Full identities and immutable
snapshots prevent false hits; the baseline and previous map can legitimately
share one entry. Adaptive expansion must extend the same paired bank, not draw
a fresh bank. Bank sizes used by this route must be whole training batches so
extension neither repeats partial batches nor checks additional, undeclared
rows. A failed validity check cannot be cached as a usable value.

Decision review: clear continuing improvement can stop precision expansion,
but neither baseline improvement alone nor a noisy near-zero increment proves
a plateau. Fine precision and the existing consecutive plateau count remain
necessary for export. Before the floor and in calibration, exports explicitly
remain ineligible. At the cap, uncertainty must remain visible even when no
further updates are possible. Tests will cover these boundary cases.

Budget review: a floor-plus-all-rungs quote mixes different execution scopes.
Use counts from the actual floor and cap separately. A warm timing does not
include every future compilation, checkpoint or state-dependent cost, and a
short primitive timing cannot fully price classical preparation, all L values
or replica exchange. The report must expose these gaps. A full campaign cannot
be declared affordable from training-only prices. The calibration command must
have its own admission and persistent incomplete status so the old guard does
not prevent the measurement needed to repair the budget.

Environment review: use an isolated checkout including the existing HMC repair;
preserve concurrent main edits and the already-built local custom operation.
Use CPU-hidden fixtures only for engineering tests. Fresh q20 timing uses
trusted GPU access with memory growth verified before initialization. No NumPy
runtime, pfor, alternate autodiff backend or package/environment changes.

Verdict: proceed. Remaining numerical hypotheses are explicit, are not promoted
by this repair, and have a bounded calibration route for examination.

## Verification and resource limit

Reserve at most 2400 seconds from the previous settled allowance of
122323.99030228473 campaign seconds, including 41347.537327354854 diagnostic
seconds. This is a convenience repair limit, not a new budget grant. Up to
1200 seconds cover focused CPU tests and localized failure repairs; up to
1100 cover fresh GPU pricing with external termination; 100 cover readiness
and accounting. Retain every attempt and charge failures. Stop before the
total limit; no full training or posterior campaign is launched by this repair.

Tests cover prefix equality/call counts, changed map/beta/seed isolation,
cache corruption, interrupted validation and exact resume, decision boundaries,
calibration-to-full-cohort continuation, nonpromotion, cost arithmetic and the
real master/worker call chain. Broaden only for failures or a newly identified
interaction. Fresh GPU pricing uses the current public training implementation
and actual q20 target at batch 32, both widths and both positive betas, two
updates and repeated heldout evaluation per scope. This is pricing evidence,
not learned-map quality evidence. If trusted GPU access is unavailable, preserve
that exact limitation and report code-derived work counts with historical
timing estimates clearly labeled.

Artifact root: `docs/plans/artifacts/ssl-lstm-q20-validation-budget-repair-2026-09-16/`.
Record actual commands, source hashes, environment, devices, seeds, wall time,
all attempts, tests, pricing receipts, scenario calculations and settlement.
Update this note with the final review and the remaining scientific limits.

## Execution and review updates

The first focused invocation passed 36 tests in 225.005655849 supervised
seconds, including real calibration workers and checkpoint reuse. Review then
identified a needless strengthening of the plateau rule: a first-bank interval
that is already precise before the floor can count as plateau evidence even
though the map cannot yet be exported as eligible. Preserve that evidence while
continuing to prohibit pre-floor and calibration promotion. Added the boundary
test, CPU/XLA cache parity and malformed pricing-metadata checks. These are
newly uncovered risks, so one additional focused invocation is justified within
the original 1200-second CPU allocation. Also run the existing full known-target
master test to exercise all downstream handoffs after calibration/runtime edits.

The trusted GPU readiness probe returned `no_idle_policy_permitted_gpu` on
September 16 at 10:33:39 UTC. All three RTX 4080 SUPER devices had active work
(observed utilization 94%, 38%, 36%). This is resource occupancy, not evidence
against TensorFlow, the GPU installation or the scientific method. No other
workload was interrupted. Pending an idle device, the reproducible budget
report uses all four batch-32 width/beta receipts from the preceding real-q20
pricing attempt and labels the result as a historical-timing estimate. It
does not issue current-source cost admission or a complete campaign forecast.

The second invocation passed 32 tests in 284.825479065 supervised seconds,
including stable-signature CPU XLA cache parity and the full known-target
master through confirmation and completed-stage replay. Across both invocations
there were 68 passing test executions (some regression tests ran in both), no
test failures and 509.831134914 supervised test seconds. Third-party TensorFlow/
gast deprecation warnings do not change these results. `git diff --check` passed.

After integration into the shared main worktree, the focused cache and cost
regression passed 19 tests in 22.59 seconds with the GPU intentionally hidden.
This final check used the exact integrated source and found no integration drift.

A second trusted readiness check at 10:41:11 UTC again found no idle permitted
GPU. No new GPU numerical run was launched. The new standalone pricing script
is ready to measure every actual batch-32 width/beta scope and cache parity
when a device is free. Its small parity bank (32 then 64 rows) tests one added
batch, not posterior coverage; its `1e-10 + 1e-9*abs(reference)` comparison
reuses the preceding q20 GPU diagnostic tolerances as a regression bound.

## Budget after the fixes

Using all four saved batch-32 q20 GPU prices, with first-call compilation
separated from steady work:

| Training scope | Optimizer updates | Validation rows | Raw hours | Reserved hours, factor two |
| --- | ---: | ---: | ---: | ---: |
| First-root, first-rung calibration, all width/LR/schedule combinations | 1,024 | 12,288 | 1.381 | 2.762 |
| All 36 scopes to floor 512, decisions resolve on first bank | 18,432 | 82,944 | 20.190 | 40.381 |
| Same floor with maximum validation expansion | 18,432 | 1,327,104 | 56.884 | 113.767 |
| All scopes to update cap 8192 and maximum validation | 294,912 | 2,211,840 | 344.903 | 689.806 |

The old 449.375-hour figure combined the optimizer floor with all four rungs'
uncached worst-case validation. It is not a valid baseline for the full-update
cap. For that old validation inventory, the new cache reduces maximum requested
rows from 6,967,296 to 2,211,840 (68.25% fewer) before decision-based early exits.
The cheaper first-bank floor is a conditional scenario, not a guaranteed cost.
The floor and cap are still protocol hypotheses, not measured necessary learning.

These figures exclude training setup/preflight absent from old timing receipts,
process and cache/checkpoint overhead, HMC tuning, posterior and confirmation
sampling, reference evaluation, and ensemble/replica-exchange work. New pricing
records setup explicitly. The full production campaign therefore remains
unpriced and cannot be declared affordable. Calibration is an approximately
2.8-hour reserved next research stage, plus current startup/pricing overhead;
it measures the learning behavior needed to revise the downstream plan. It
does not certify sufficient training or select a winner from one root.

Machine-readable and rendered arithmetic:
`artifacts/ssl-lstm-q20-validation-budget-repair-2026-09-16/budget-historical-estimate/budget.json`
and the adjacent `budget.md` preserve the initial estimate. The final report
under `budget-after-repair/budget.md` uses the settled remaining allowance.
Historical price paths and checksums are retained. Five selected numerical
target/trainer modules match the historical pricing manifest exactly, whose
GPU memory-growth record is valid; the enclosing cache/decision changes still
require new timing before current-source admission.

Settlement charges 509.886391457 measured test/readiness seconds plus the
predeclared 100-second artifact/accounting hold: 609.886391457 seconds total.
The unused 1790.113608543 seconds of this phase's 2400-second reservation are
released; no allowance is renewed. Remaining campaign time is 33.80947 hours,
including 11.31601 diagnostic hours. Both readiness failures are recorded as
resource-availability checks, not GPU numerical attempts.

## Terminal review and decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept validation-cache and scheduler repair | Equal prefix losses, exact optimizer continuation, boundary decisions and executable master tests passed | No observed corruption acceptance, replay drift or early promotion | New GPU cache cost and resource use are unmeasured | Fresh bounded GPU pricing when idle, then explicit calibration stage | Learning adequacy, whitening or posterior convergence |
| Accept corrected training work counts as planning arithmetic | Counts match distinct frozen maps, nested prefixes, actual floor and cap | Missing costs are explicit; no complete campaign admission | Learning may need more/fewer updates or expanded validation | Inspect calibration learning/variance; revise hypotheses before full commitment | 40.381 hours is a guaranteed requirement or bound |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Engineering and known-target checks passed; real q20 GPU run not attempted because devices were occupied |
| Statistically supported ranking | None; calibration retains all declared combinations |
| Descriptive-only differences | Historical per-batch times applied to new work counts |
| Default-readiness | Repaired execution is available; numerical defaults remain hypotheses, scientific readiness unestablished |
| Next evidence needed | Current GPU costs, target-specific learning/variance curves, full seed replication and downstream posterior/reference checks |

Post-run red-team: the strongest alternative explanation for a future slow run
is the strict 80-dimensional factor/derivative target itself. Cache savings
remove redundant work without accelerating each unique target call. Another
uncertainty is state-dependent cost after substantial learning, which short
pricing cannot bound. The smallest overturning evidence would be failure of
real-q20 cached/uncached parity or a materially different current per-batch cost;
the fresh pricing script preserves both. No HMC health telemetry is removed,
no target backend is substituted, and no relaxed posterior criterion funds the
smaller calibration stage.

## Recovery and next commands

The shared main worktree receives only this repair's files; existing batched-HMC
and concurrent changes remain. There is no new Git commit or long q20 training
launch in this repair. The calibration command is:

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py calibrate --output-dir <fresh-versioned-root> --budget-record docs/plans/artifacts/ssl-lstm-q20-validation-budget-repair-2026-09-16/settled-allowance.json
```

Run with trusted GPU permissions and `TF_FORCE_GPU_ALLOW_GROWTH=true`; the
coordinator selects an idle permitted device and workers verify memory growth.
The completed calibration can be continued with `campaign` against the same
output root and exact source/configuration. That continuation still performs
full qualification, pricing and affordability checks before extending research.
Do not reuse this old-timing report as a current pricing receipt.
