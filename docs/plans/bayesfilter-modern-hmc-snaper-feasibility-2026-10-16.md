# Frozen SNAPER-proposal feasibility — 16 October 2026

This continues the October 7 modern-HMC master after the completed randomized
cohort. It consumes the already-frozen target-specific discarded SNAPER pilot
proposals through the public ordinary candidate-set tuner. It is not a fresh
replication of adaptation. NUTS remains excluded.

**Terminal update, 17 October:** The cohort finished at 10:21 China time
and its audit closed at 10:22. All six roots qualified and were assessed;
five deliveries passed posterior and reference screens. Member 1 failed the
declared lugsail precision limit at the retained-count cap, while passing
its reference screen. No repair or numerical-health veto occurred. All six
remain tuning-qualified. The [terminal result](bayesfilter-modern-hmc-snaper-terminal-2026-10-17.md)
records authoritative posterior rows, integrity checks and the reconciled
`137294.5091979513`-second net release. The release is separate from running
SIR caps. The plan below preserves the original pre-run contract and costs;
its launch observations are historical, not a current running status.

## Research intent and evidence contract

The question is whether all six frozen SNAPER proposals and any declared
repair descendants receive valid fresh qualification and complete posterior
dispositions on full NL-PP-SGQF. The exact comparator ladder retains the
completed fixed-HMC and randomized-HMC cohorts and fully refreshed MAMS,
using the same immutable numerical source, target, data, geometry and
posterior/reference rules. The simple fixed-length, uniform-duration and
fully refreshed proposals are the constructed classical comparators; the
finite-filter reference supplies the independent target check. This phase
does not establish conditional heuristic dominance across models or regimes.

The primary engineering criterion is a complete, auditable candidate
denominator and a disposition for every qualified member. Acceptance lower
bounds and numerical health determine tuning membership. All qualified
members are retained. The frozen posterior R-hat, ESS, lugsail MCSE and
18-quantity reference screen determine posterior delivery only. They cannot
reject, rank, repair or delay a tuning candidate. Passing establishes
feasibility for this target and seed; it establishes no method ranking,
stopping-interval coverage, adaptation reliability, general posterior validity,
new default or release readiness.

Invalid source/target/data identity, corrupted checkpoints, missing essential
diagnostics, or invalid GPU allocation veto affected evidence and trigger
infrastructure investigation. Shared corruption and exhausted allocation are
continuation stops. A candidate failure is a delivery or tuning outcome under
the frozen policy, not a stop for unaffected peers or the research direction.
Wall-time exhaustion is resource censoring. Prices, acceptance preferences,
throughput and progress counts are explanatory only.

## Exact frozen scope and numerical provenance

Artifact base `C` is
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01`.
New output is `C/snaper-feasibility-01`. Source `S` is the sibling
`generic-adaptation/full-model-source-e68cdf1c4`, commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`.

The design is unchanged `C/designs-02/snaper.json`, SHA-256
`4d82f5998c84b3b1bf21b56c5effbc51bcc9feb1f0ac83dd544b493fdc57ce33`.
Its six `(maximum steps, epsilon)` roots are
`(2,.6269862024585631), (2,.7837327530732038),
(4,.6269862024585631), (4,.7837327530732038),
(7,.7837327530732038), (8,.6269862024585631)`.
They come from the saved target-specific discarded pilot, SHA-256
`64c1fa630ce7a84f1159c2d9c3743661421deb004c4f9eb8ba25155f08aa9e41`.
No prior tuning qualification or posterior draw is reused.

| Choice | Provenance and status | Failure mode and earliest diagnostic |
| --- | --- | --- |
| Six roots, maximum 12 candidates, one repair per family, preserved mean duration | Frozen pilot-derived hypotheses | Pilot overfit or unsuitable duration; independent qualification and complete root/descendant inventory |
| Seed 71410; explicit execution/posterior streams | Frozen comparison design | One seed cannot establish reliability; audit identities and disjoint streams |
| v8 lower-bound policy; 32/64/128 independent trials of length 64 | Frozen experimental policy | Insufficient power or dependence; actual-source attainability preflight and saved evidence |
| Four chains; 2000 minimum warm-up, 10000 maximum warm-up and retained draws | Frozen comparison baseline, not universal adequacy | Count-limited precision failure; sequential diagnostics with failures preserved |
| R-hat 1.05 warm-up/1.01 retained; ESS 400; MCSE/SD .02; original lugsail r=3,c=.5,min_batches=20 | Inherited comparison policy with limited calibration; optional safeguard study retained separately | Sequential coverage or variance underestimation; separate reference screen, no retrospective estimator change |
| Reference normalization .1 with MCSE multiplier 3, 18 quantities | Frozen finite-filter reference contract | Approximation/reference error or uncertain MCSE; compatibility audit and all-quantity results |
| GPU/XLA, TF32, memory growth, two framework/OMP threads | Existing source and hardware-class contract | Contention or initialization drift; trusted probe and actual-source preflight |
| 180000-second inclusive cap | Resource scheduling choice: round the old 176233-second minimum-path scenario up to 49 hours, plus one hour; 50 hours total | Forecasts are not bounds; checkpoints and explicit resource censoring |

The old SNAPER pricing scenarios at load multipliers 1/2/4 are respectively
176233/331486/641993 seconds for the minimum path and
878957/1675002/3267093 seconds for the maximum-count path. These are descriptive
scenarios based on old measurements, not statistical bounds or continuation
vetoes. The preceding randomized cohort finished much faster than its old
forecast. Completion within this new ceiling is not promised.

## Accounting, execution and recovery

The old randomized settlement released 69278.81444643158 seconds. The owner
grants total 345600 seconds; after the recorded preparation and terminal
observer charges, they retain 345499.4271640065 seconds. Their reconciled
combined pool is 414778.2416104381 seconds before this phase's measured
preparation. The old 3600-second reserve is already held and the separate
1579.5929542018566-second historical balance is excluded. No overlapping
supervisor/child double charge or repeated settlement is permitted.

Reserve 180000 seconds for one numerical launch, leaving roughly 234778
seconds before measured preparation/observer costs for subsequent work. The
allocation records the exact reconciled value after preflight. A localized
failure may be repaired and retried in a new versioned directory within the
remaining pool and unchanged scientific contract; every attempt is charged.
There is no unattended numerical retry or automatic follow-on launch.

1. Record the randomized terminal result and reconcile existing receipts once.
2. Repair the read-only settlement script's hardcoded old-grant label and plan
   path. Validate optional funding-source balances and preserve legacy inputs;
   test mixed pools, malformed balances and terminal observer plan provenance.
   This changes no numerical sampler code. Freeze a new observer copy.
3. Check the exact design against the fixed/randomized posterior and execution
   contracts, allowing declared method, proposal inventory, seed-stream and
   statistical-family allowance differences. Verify the pilot hash. Run the
   actual immutable-source binding/reference/attainability preflight on a
   trusted GPU with growth configured before TensorFlow import; no sampling.
4. Launch detached through systemd with a fresh output directory. The command is
   `S/scripts/run_modern_hmc_frozen_full_model.py --design C/designs-02/snaper.json
   --output C/snaper-feasibility-01/run --max-wall-seconds 180000`, using
   `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. Record exact absolute arguments,
   GPU UUID/index, environment, source/design hashes and command wall times.
5. Install bounded read-only progress and terminal-audit observers. At exit,
   audit every root/descendant/member, source/input hashes and saved tensors;
   settle inclusive supervisor time once, then observer active time separately.
6. If resource-censored, retain all completed outcomes. Posterior continuation
   is available after complete tuning; incomplete-qualification recovery must
   first be validated against the public checkpoint interface. Never restart
   qualification silently. Count-limited or reference-failed deliveries remain
   terminal. Then continue the master with justified model/seed and N8 work.

## Skeptical pre-execution audit

**Pass for this bounded feasibility phase, conditional on successful runtime
preflight.** The review found a real reporting flaw: the closeout script
hardcodes exclusion of new grants, so a combined allocation would be reported
incorrectly. Repair and regression tests precede its deployment. A stale
progress-observer time bound must also match the new cap. Numerical source
and qualification/posterior rules remain frozen. Actual consumer wiring uses
`run_frozen_trajectory_search` and the public ordinary tuner, as documented by
the capability registry; the preflight checks the bound call path and policies.

The main misleading success would be presenting saved pilot proposals as
independent adaptation or treating five versus seven past deliveries as a
method ranking. Explicit scope and complete outcomes prevent those claims.
A resource timeout or single candidate failure could misleadingly reject
SNAPER; recording censoring, preserving peers and using the stated recovery
route separates these cases. A launch command that changed the target,
precision method, data or count caps would not answer this question and must
fail preflight. CPU-only/eager execution cannot silently substitute for the
GPU/XLA path. Existing stopped-precision coverage and broader model validation
remain open master-program tasks, not hidden claims of this run.

## Execution record, 16 October, 22:34 China time

The skeptical audit's preparation conditions passed. Design comparison
confirmed unchanged target/data/geometry and posterior policy; only declared
proposal, seed-stream and candidate-allocation fields differ. All 90 focused
settlement/observer regressions passed. Actual-source GPU preflight completed
in 363.31657919473946 occupied seconds; it checked all six roots, the pilot
hash, qualification attainability, reference compatibility and growth.
Combined new preparation cost was 363.9916459270753 seconds.

The worker launched on GPU 1 at 22:30 China time as
`bayesfilter-modern-snaper-feasibility-20261016-01.service`. Both read-only
observers are active with zero restarts. The worker is currently preparing;
no SNAPER qualification or posterior result is claimed. `launch.json`,
`launch-receipt.json`, `allocation.json`, `execution-note.json`, and the live
`run/manifest.json` preserve exact commands, source/data/geometry, GPU UUID,
memory policy, environment and seed streams. The latest GPU capacity probe
passed after the preflight released the device.

The 180000-second reservation leaves 234414.24996451102 seconds unallocated
before eventual observer costs. Existing reserves remain held separately.
The nominal ceiling is about 19 October at 00:30 China time, not an ETA.
The new observer reports explicit funding sources and this phase plan; it
does not repeat the previous cohort settlement. Terminal results will be
reported with the full denominator and separate tuning/posterior decisions.

At 22:34 China time the worker advanced to `qualifying`, with no reported
error. No tuning or posterior outcome is available yet.

At 17 October 02:52 China time, tuning is complete: all six roots qualified,
no repair descendants, 480 saved qualification numerical chunks. Posterior
member 000 is running and the other five are pending; no final delivery is
claimed. All three services remain active with zero restarts, and source,
binding and memory-growth checks pass. The parallel CPU-only N8 engineering
slice completed in 522.8989839983948 occupied seconds, charged to unallocated
funds separately; `N8/consumer-audit-20261017-01/accounting-01.json` records
233891.35098051262 seconds unallocated with this numerical reservation intact.
The existing terminal observer, not the N8 accounting, settles this cohort.
