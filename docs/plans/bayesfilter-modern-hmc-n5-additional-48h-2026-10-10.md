# N5 continuation under the additional 48-hour grant

This supplements the [N5 campaign](bayesfilter-modern-hmc-n5-campaign-2026-10-08.md)
and [master](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md).
**Terminal-stage refresh:** D2f, D3 and E3 are complete. The active continuation
is the [concrete D4 plan](bayesfilter-modern-hmc-n5-method-proposals-2026-10-10.md),
with the settled ledger and twenty-cell inventory in `N5/method-study-refresh-01/`.
The initial grant/allocation numbers below are historical starting values;
193403.4474935018 seconds are now charged, leaving 66.11 hours after reserve.
All fifteen structural reference quantities pass the unchanged empirical screen.
No sampler tuning or posterior qualification is implied by that reference result.
Subsequent D4 accounting, including focused tests and the stopped setup attempt,
is recorded in `N5/method-study-refresh-01/budget-ledger-d4-retry-baseline.json`;
the then-active retry and observer were `N5/method-proposals-02/` and
`N5/method-study-watch-02/`. Per-cell automatic trusted GPU selection replaces
the initial fixed-device assumption. No additional grant was needed.
That retry is now terminal: all eight cells and sixty native paths finished.
Active D4b1 uses `N5/method-full-fit-01/`, observer
`N5/method-full-fit-watch-01/`, and baseline
`N5/method-full-fit-audit-01/budget-ledger-d4b-baseline.json`. These newer records
supersede the earlier active-status wording; historical costs remain charged.
The owner's October 10 request adds 172,800 aggregate numerical-worker seconds
to the previous 259,200-second grant. The cumulative ceiling is 432,000 seconds.
The settled 180,361.9048404594 seconds remain charged. Before new work, there
are 251,038.0951595406 unallocated seconds after the existing 600-second reserve.
Parallel worker durations add; nested children are not charged twice. Tests,
failed launches and failed candidates consume the same budget. Passive resource
waiting is recorded separately. No new approval-token or review chain is needed.

## Research intent and evidence contract

Question: can the existing experimental modern samplers deliver separately
assessed posterior estimates on the full nonlinear state-space fixtures under
the unchanged target, reference and precision checks? Fixed and randomized HMC
are the classical comparators. SNAPER is discarded trajectory adaptation followed
by fresh ordinary tuning; MAMS uses its existing corrected full-refresh kernel;
NUTS is excluded by the owner, including diagnostic comparators. The
[scope repair](bayesfilter-nuts-exclusion-repair-2026-10-10.md) supersedes the
older nine-row price inventory. Only six SNAPER/MAMS price cells are required.

Tuning qualification uses the public tuner, explicit replicated acceptance policy
and numerical health, never R-hat, ESS, MCSE or reference agreement. Posterior
delivery requires the declared warm-up, retained R-hat, bulk/tail ESS, lugsail
precision and independent agreement checks. These are posterior pass criteria
and promotion vetoes only. A precision-limited candidate triggers a predeclared
duration/geometry study; it does not invalidate its tuning membership or veto
the research direction. Numerical non-finiteness, target/status failure,
corrupt identity/checkpoint, changed source or data and invalid reference are
continuation vetoes for the affected run. Wall-budget exhaustion preserves
partial evidence and triggers a resource diagnosis. Runtime, acceptance and
short trajectories are explanatory diagnostics, not comparative superiority.

Preserve all original candidates, failed attempts and denominators. Passing a
screen establishes neither exact nonlinear-posterior accuracy nor release/default
readiness. One-seed comparisons are descriptive; ranking needs a fresh paired
replication plan and uncertainty evidence. Original interrupted fits must not be
reconstructed as complete from progress. Results live in fresh directories under
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/` and are recorded in
the existing execution note and comparison matrix.

## Ordered phases and bounded allocations

1. **D2f, checkpoint continuation.** Add a consumer of the existing retained
   loader, shared posterior controller and durable checkpoint authority. Validate
   source, member, reference, policy, seed and GPU identity; copy the inactive
   original store under its writer lock to `ukf-member-continuation-01/`.
   Replay committed chunks without numerical recomputation, then run only UKF
   randomized L=9. Do not retune, rerun L=3 or alter counts/precision thresholds.
   Reserve 36,000 seconds for supervisor/worker and 120 for independent assessment.
   Preserve interrupted outcomes on timeout and record replayed/new chunks.
   Test real native replay equivalence, changed policy/seed/source rejection,
   corruption rejection, source immutability and process-group timeout cleanup.
2. **D3, full nonlinear comparator consumers and pricing.** Add separate harnesses
   for UKF/SGQF/SIR SNAPER/MAMS using existing numerical authorities, full data,
   identical frozen affine geometry/starts and declared model quantities. Do not
   modify old source closures. First test lifecycle and target wiring, then reserve
   at most 24,000 seconds for six short, individually bounded pricing cells.
   Keep every cell, including pilot/tuning failures. Pricing nominates budgets;
   it does not certify convergence. Freeze target-specific epsilon/pilot
   hypotheses before a subsequent full-fit phase, using fresh tuning evidence.
3. **E3, structural reference repair.** Use fresh calibration draws to propose
   quantity-influence-aware coverage repair, with the original fixed prior
   component and exact mixture denominator. Check on fresh untouched scrambles;
   keep all 15 quantities and the unchanged .002-SD reference criteria. Proposed
   ceiling 36,000 CPU-worker seconds, but reserve only after a concrete proposal,
   independent Gaussian controls and price check pass. No sampler-target or
   geometry changes and no reuse of selected validation replays as replications.
4. **D4, refreshed full-fit/precision study.** After D3/E3, write the smallest
   discriminating next phase with measured cost and remaining aggregate budget.
   Do not enlarge counts blindly or select a tuning member by posterior metrics.
   At most two campaign GPU workers; refresh plan, result matrix and budget after
   each terminal phase. Release unused caps once. Stop when the campaign ceiling
   is exhausted or a true scientific continuation veto cannot be repaired.

Only D2f is reserved initially. Subsequent allocations are ceilings, not charges
or an obligation to spend. The remaining budget is available for full-fit
validation after prices and references justify it.

## Default, numerical-provenance and assumption audit

| Choice | Provenance and role | Failure mode and early diagnostic |
| --- | --- | --- |
| Aggregate grant/accounting | Inherited interpretation of the 72-hour grant plus the explicit new 48-hour grant | Double charging nested processes; retain prior ledger and add overlapping workers |
| D2f 36,000-second cap | Convenience hypothesis inherited from the SIR full-fit cap, not a guaranteed UKF forecast | Contention/compile variation; track chunk durations, preserve checkpoints and stop at cap |
| Original 1,000-draw chunks, 2,000–10,000 warm-up and 10,000 retained cap | Exact saved UKF policy, unchanged baseline | Precision can still fail; do not call timeout/precision failure a tuner defect |
| R-hat/ESS/.02 MCSE ratio/.002 reference criteria | Existing declared operational screens, not universal guarantees | Repeated-look uncertainty/limited reference; retain interpretations and all quantity results |
| GPU 1, float64, GPU/XLA and memory growth | Original device/policy; shared numerical authorities | Wrong device or allocator; trusted inventory and verified growth before initialization |
| Frozen local affine geometry | Existing target-specific proposal, not learned NeuTra | Residual curvature; compare identical starts and record bounded failures |
| D3 24,000/E3 36,000 ceilings | Convenience planning hypotheses | Insufficient full-fit evidence; price first and refresh rather than assert readiness |
| New calibration/validation seeds | Explicit fresh partitions, recorded before launch | Accidental reuse/selection bias; manifest partition roles and never fit validation draws |

## Skeptical pre-execution review

The wrong baseline would be a rerun that drops the L=3 failure or changes the
kernel after interruption. D2f instead preserves the original two-member inventory
and uses exact existing checkpoint identities. Loading a committed chunk is not
a new replication. Source closures must stay untouched; new consumers can reuse
existing numerical authorities without invalidating old evidence. A copied store
must reject corrupt committed data rather than regenerate it. Tests must exercise
the actual native kernel/shared controller, not just mocked record counts.

A short price, acceptance or posterior pass cannot support a performance ranking.
Full nonlinear references evaluate the same finite filter target, not the exact
latent-state posterior. E3 must not fit held-out draws. Warm-up stays excluded;
the old L=3 results remain terminal. Device availability alone does not imply
throughput, and concurrency adds cost. The plan has explicit attempt ceilings,
terminal manifests and process-group/cgroup stop conditions. No material scientific
flaw remains in D2f's design; proceed after its focused regressions. D3/E3 must
receive their concrete assumption/price review before numerical launch.

## D2f launch and D3 scope correction

Eight CPU/non-XLA regressions passed in 48.84 pytest seconds (50.786 measured
subprocess seconds). Native interrupted-prefix continuation equals uninterrupted
same-seed draws, skips the committed computation and leaves the source store
unchanged. Changed seeds/counts/member/quantities, corrupt tensors, a live writer
and an existing output are rejected; timeout cleanup is bounded. Saved source
and reference preflight passed. D2f launched at 17:55:58 UTC on October 9
(01:55:58 China time on October 10), on the original physical GPU 1, with verified
memory growth and GPU/XLA. Progress shows chunk 000000 replayed and chunk 000001
executing. The original first chunk's 2,211.83-second cost is historical and is
not charged again as new numerical computation. Actual replay/validation wall
time is charged. `campaign-120h-audit/budget-ledger-01.json` records the grant,
test cost and 36,140-second reservation.

The first D3 review incorrectly treated NUTS as a supported research direction
with an internal-status implementation gap. The owner corrected that scope:
NUTS is excluded and no extension is planned. The six SNAPER/MAMS cells retain
checked native `per_chain_step` status coverage. The old launched manifest has
three nonexecuting NUTS bookkeeping rows; preserve it as run history and use the
current scope overlay for required coverage. No NUTS worker was launched by D3.

## D3 concrete price contract and pre-run review

`scripts/price_modern_hmc_full_comparators.py` freezes the same target-specific
geometry, four starts, complete observations and per-substep status policy as
the classical consumer. The six supported cells use seed 4200 and fresh stateless
streams `[20261010, 5200+i]`. Each frozen short trajectory has 32 transitions
per chain. MAMS starts with `(L, epsilon)=(3,.2),(9,.2)`: these are transferred
HMC hypotheses, not tuned MAMS settings. SNAPER's 32-step pilot starts at .2,
with .8 acceptance target, maximum 32 steps, no initial MALA, learning rate .05,
EMA factors 8 and epsilon factors .8/1. These are inherited small-target pilot
settings except the initial epsilon, which comes from the full-target HMC price.
The early diagnostic is finite pilot/state/status and checked native short
trajectories; failure does not reject the research idea. All pilot draws are
discarded and starts reset. Every proposed pair requires fresh public tuning
before posterior execution. Pilot status covers accepted pilot states only;
the frozen native price restores complete internal-step status coverage.

Six numerical cells each have a 3,600-second convenience price cap inside a
24,000-second queue ceiling. The primary engineering criterion is complete,
hash-checked native trace/work and health records; nonfinite/filter-status
failure vetoes a price candidate. Reference validity and source/data integrity
veto execution. Runtime, acceptance and proposal length are explanatory prices,
not tuning qualification, posterior validity or ranking. All six approved cells
remain required; old excluded bookkeeping rows are retained only as history. GPU 0 may run this independent queue only after GPU 1
is occupied by D2f and a fresh trusted auto/device probe admits GPU 0.

The initial full-horizon CPU/non-XLA regression cap of 240 seconds is a
convenience hypothesis and is not evidence that these eager reference models
should fit within four minutes. Preserve a capped attempt and its charge if it
expires. A single diagnostic rerun may use a bounded 1,800-second allowance;
these are full-target mechanics tests, not a CPU production recommendation.

Execution correction: the CPU rerun was interrupted for harness/environment
diagnosis during full-model setup; it did not establish a mathematical failure.
Its full 1,800-second cap is conservatively charged because the exact interrupted
subprocess duration was not recorded. Full-horizon integration is now opt-in,
and this campaign will run it on trusted GPU 0 with verified growth and XLA,
under the same 1,800-second test ceiling. This avoids treating slow eager CPU
setup as a requirement for the default GPU path. Lightweight denominator and
accounting checks remain in the ordinary regression suite. Full-horizon model,
data, geometry and status assertions are retained in the explicit GPU integration
run; no numerical horizon or correctness check is reduced.

Skeptical review: do not compare MAMS at an HMC-transferred step as if it were
tuned or treat a 32-step pilot as convergence. Preserve excluded historical
NUTS outcomes and costs separately from current scope. The queue
records raw durable chunks, all proposals, failures and source identity. Full
model/geometry tests and actual native MAMS mechanics precede the price launch.
No numerical authority is copied or replaced. Price outcomes only inform the
next audited full-fit allocation, so the six-cell price is appropriate after
the focused regressions pass.

## E3 concrete proposal repair and pre-run review

The old reference is precise for fourteen quantities but not source-square rho.
Selected replay points diagnosed influence and are not independent replications.
The E3 question is whether a quantity-aware proposal can meet the same precision
criterion with fewer validation points than a blind increase to the old order.
The comparator is the preserved eight-component frozen reference proposal and
its failed power-21/23 validation. It is a descriptive historical reference,
not a paired new performance comparison.

For unnormalized target density p, normalizer Z and a mean mu_j of h_j,
the self-normalized importance estimator's asymptotic mean variance is
`Z^-2 integral p(q)^2 (h_j(q)-mu_j)^2 / g(q) dq`, under its CLT assumptions.
By Cauchy–Schwarz, minimizing a weighted sum of these integrals over unrestricted
normalized g gives `g proportional to p sqrt(sum_j a_j (h_j-mu_j)^2)`.
We add a constant normalizer term and use equal standardized quantity weights:
`envelope=sqrt(1 + mean_j ((h_j-mu_j)/sd_j)^2)` for all fifteen declared quantities.
Fresh calibration estimates mu/sd; weighted EM receives posterior fitting weights
multiplied by this envelope. The final reference still uses **p/g**, with the
complete exact equal-mixture denominator. Fitting weights are never inference
weights. The fitted Gaussian mixture is a heuristic approximation to this
variance-oriented density, not an optimal proposal or a correctness proof.

Warm-start from the old frozen proposal, preserving its fixed N(0,I) prior
component. Use two fresh power-15 calibration rounds, seeds 21261010 and 22261010,
then freeze the proposal before untouched eight-scramble validation with seed
230361010 and powers 19/21. Per-component seed ranges are disjoint from both
calibration rounds and the old validation. Twelve weighted-EM updates and .1
global-covariance shrinkage are inherited proposal-only hypotheses; no target
regularization changes. Equal standardization and the constant term are newly
derived design hypotheses. Calibration scale failure, nonfinite weights or
nonpositive-definite factors reject the proposal; no quantity is silently omitted.

Primary pass criterion: all fifteen quantities satisfy the unchanged .002-SD
pointwise half-width and last-order change screens, with native/independent
value-score parity and complete provenance. Reference precision failure vetoes
posterior-reference qualification, not the target or sampler direction. The
estimated normalizer, importance-weight ESS, maximum weights, tail influence and
runtime are explanatory only. The reference is for the same finite UKF posterior;
there is no exact latent-model posterior, rigorous quadrature bound or ranking
claim. Artifacts live at `N5/structural-influence-reference-01/`.

Before launch, test the influence algebra against an independent calculation,
invalid/constant quantity rejection, preservation of the prior component and
unchanged inference weights, and two fresh-round receipt/seed separation. A
Gaussian control checks the unchanged full-mixture denominator. Reserve only
12,000 CPU-worker seconds for this first concrete attempt, within E3's proposed
36,000 ceiling; calibrate first and preserve each validation replication. Further
orders require terminal price/precision review and remaining-budget allocation.

Skeptical review: empirical fitting may miss a rare tail even with a defensive
prior component; that is tested by untouched scrambles and quantity-level
uncertainty rather than weight ESS alone. The prior guarantees support, not
finite variance or reliable tail estimation. Calibration uncertainty and the
restricted mixture can defeat the heuristic; the unchanged screens must remain.
The new diagnostic module reuses the existing independent density, quantities,
weighted EM and validation engine; old closures are untouched. No validation
draws are fitted and no selected replay becomes a replication. This is a bounded
proposal repair with an unchanged inference target, so proceed after focused
controls and explicit source/input preflight.

E3's fifteen focused reference controls passed in 2.63 pytest seconds. Original
source/input checks passed. The bounded CPU attempt launched at 18:48:49 UTC,
completed both calibration rounds in 12.239 seconds, froze the proposal and
completed the first power-19 validation order. Power 21 is in progress. Final
reference precision remains unassessed. D3's full GPU/XLA regression has passed
the SGQF, UKF and SIR data/geometry/status checks; the final native MAMS check
is running. Four queue/passive-wait-accounting controls pass. The staged queue
starts the six numerical price cells only after that regression returns zero.
Fresh device readiness remains required by the existing launcher for each cell.

Live accounting and alerts are in `N5/supplement-watch-02/`, recomputed from
`N5/campaign-120h-audit/budget-ledger-03.json`. Its reservations include D2f,
the explicit GPU regression, E3 and D3 prices once each; the nested numerical
children are not separate charges. The previous observer is superseded and
preserved. About 48.56 worker-hours remain unallocated after these reservations
and the original overhead reserve; this is not a forecast of campaign completion.

The next phase refresh must also address eager scalar binding probes: source
inspection shows `_probe` evaluates four scalar starts eagerly even in a GPU/XLA
binding, and preparation consumers construct multiple bindings. The slow
full-horizon test exposed this engineering cost. Do not edit any bound numerical
source while the saved-member/reference runs are active. After terminal
assessment, price a single shared compiled value/score/status preparation route
with explicit stable signatures, full-model parity and source-scope handling.
This is a setup-cost repair, not a changed target or relaxed tuning criterion.

## D2f terminal review and D3 launch recovery

D2f finished at 19:06:50 UTC. The resumed L=9 member passed posterior and matched
reference checks after 2,000 warm-up and 5,000 retained draws per chain; its
hard-veto list is empty. The worst unchanged .10 reference agreement screen is
.082159. One original committed chunk was replayed and six new chunks computed.
All warm-up draws remain excluded. The original store is unchanged, the L=3
precision failure remains terminal, and the original fit result is still absent.
This is a separate member-continuation result, not an all-member fit pass.
The complete additional cost, including assessment, is 4,252.341 worker seconds.
Across the original eight UKF/SIR candidates, separately combining this result
with preserved outcomes gives five posterior/reference passes; one of four fits
passes every member. These are descriptive counts, not reliability or ranking.

All six full-model GPU/XLA regressions passed in 1,398.899 subprocess seconds.
The first price launch then correctly rejected GPU 0 because D2f had finished
and GPU 1 was available. No numerical price ran; its .318 occupied seconds and
900.039 passive waiting seconds are recorded separately. After a fresh trusted
probe, the unchanged price program launched on GPU 1 at 19:11:35 UTC into
`N5/full-comparator-prices-02/`. This is a resource allocation repair, within the
same hardware class, target/method/seeds/checks and budget. The old failed
manifest remains intact. E3 continues untouched validation on CPU.

The terminal audit is `N5/campaign-120h-audit/d2f-terminal-audit.json`; the current
matrix is `nonlinear-comparison-matrix-02.json` in that directory. The settled
reservation baseline is `budget-ledger-04.json`, monitored by
`N5/supplement-watch-03/`. Previous observers are superseded, not erased.
At refresh, 52.67 worker-hours were charged including active E3 elapsed time,
9.63 reserved and 57.53 unallocated after the existing overhead reserve. These
quantities change as workers finish; the live ledger is authoritative for the
next allocation.

Terminal review: D2f establishes scoped delivery of that one verified member.
It does not rescue L=3 precision, complete the original interrupted fit, prove
mixing generally or rank methods. Reduced contention remains an alternative
explanation for lower observed chunk costs. E3 and D3 must reach their own
terminal scientific assessments before D4's measured-cost/full-fit refresh.
