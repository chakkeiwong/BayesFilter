# PP-UKF Operational Broad-Grid Tuning Plan

Date: 2026-07-21

Status: `EXECUTED_TERMINAL_VIABLE_PAIR_SET`

Terminal result:
`docs/plans/bayesfilter-pp-ukf-operational-broad-grid-tuning-result-2026-07-21.md`

## Objective

Run the fixed-identity PP-UKF NeuTra target through the reviewed operational
broad-grid policy: independently tune epsilon at
`L=(3,5,9,13,18,25)`, preserve every viable primary pair, then evaluate each
viable primary's one-hop `L-1` and `L+1` neighbors at the exact same epsilon.
This replaces the wrong anchor-local/bootstrap-only PP-UKF tuning attempt for
this campaign; it does not silently change the generic public tuner default.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Does scope-specific independent epsilon tuning over the complete six-point L grid yield any viable PP-UKF fixed-identity HMC pairs, and are those pairs locally suitable under same-epsilon neighbor guards? |
| Candidate | Frozen PP-UKF NeuTra target, identity metric, each `(L, epsilon_L)` primary and its same-epsilon one-hop guards |
| Expected failure mode | A shared epsilon/bootstrap anchor misses viable L-specific scales; large-L arms may be expensive; short adaptation may not stabilize epsilon |
| Promotion criterion | Complete six-primary barrier plus complete guards for every viable primary, with at least one provisionally viable pair and no hard target/lineage/resource veto |
| Promotion veto | Target/transport/metric/coordinate drift, nonfinite state/value/score/log acceptance, invalid target status, positive exposed divergence, failed movement, or incomplete barrier |
| Continuation veto | Fresh cost projection exceeds the remaining four-GPU-hour campaign ceiling; GPU memory policy fails; shared harness/target invalidity; artifact corruption |
| Repair trigger | A candidate-local rejection preserves other primaries; unstable epsilon triggers a later reviewed adaptation-budget repair, not transfer from another L |
| Explanatory diagnostics | Per-arm epsilon, acceptance means and intervals, movement, status, compile/run time, and allocator telemetry |
| Must not be concluded | No candidate is statistically best; no posterior convergence, retained-sampling readiness, sampler superiority, target correctness, or default readiness follows |

## Evidence Contract

- Exact comparator: the prior PP-UKF public-tuner canary on the same target and
  transport, which executed only `L=(5,3,3,3,3,3)` during bootstrap and never
  reached a broad L grid.
- Target signature:
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Metric: fixed identity in frozen-transport coordinates; no mass adaptation.
- Primaries: exactly `L=(3,5,9,13,18,25)`. Epsilon is dual-averaged
  independently for every primary.
- Guards: exactly one hop from viable primaries, bounded to `2 <= L <= 25`,
  using the parent's bit-identical epsilon with no retuning or recursion.
- Evidence: four chains and three fresh replications per pair. The scheduler's
  working interval is a bounded tuning heuristic, not a confidence guarantee.
- Artifact root:
  `docs/plans/artifacts/bayesfilter-pp-ukf-operational-broad-grid-20260721/`.

## Frozen Mechanics And Budget Ladder

1. Use TensorFlow/TFP, GPU, XLA, float64, TF32 recorded, and verified memory
   growth. Use the existing frozen transport and current batch-native combined
   value/score/status PP-UKF target.
2. Use the same four-chain start bank as frozen validation: row offsets
   `(0.0, 0.1, -0.1, 0.16)` broadcast over six parameters.
3. Warm-start dual averaging at epsilon `0.9853849721883557`, the last finite
   PP-UKF bootstrap diagnostic value recorded in
   `docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-01/PP-UKF/result.json`.
   It is a warm start only, never a default or promotion fact. Use 64
   burn-in/adaptation transitions and one discarded result for each primary.
4. Screen with three fresh replications, 128 discarded results per chain and
   one initialization transition. Target acceptance is 0.70, practical region
   `[0.65,0.75]`, repair region `[0.55,0.85]`.
5. First run one complete `L=3` primary canary. Project the remaining five
   primaries from measured leapfrog work with a 50% margin and reserve all prior
   charged campaign time. If the primary-only projection exceeds four GPU-hours
   total, stop before the full primary barrier. A partial primary grid cannot be
   promoted.
6. If the primary gate passes, run all six primaries serially in one GPU process
   so the reusable dynamic-L XLA graphs amortize compilation. Only after the
   primary barrier is complete, expand the actual guards from provisionally
   viable primaries and project that exact guard set with the same margin. Run
   all actual guards only if the cumulative projection remains within the cap.
   Every draw is discarded.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Six-point L grid | Reviewed `hmc_operational_broad_grid` policy | Grid may still miss useful L | Preserve no-ranking/non-exhaustiveness nonclaim |
| Independent epsilon per L | User-confirmed algorithm and reviewed policy | Adaptation too short for an arm | Finite scalar final epsilon plus fresh screens |
| Same-epsilon neighbors | Reviewed isolation guard | Retuned neighbors would answer a different question | Payload must state `epsilon_retuned=false` |
| Fixed identity | Existing PP-UKF campaign scope | Poor geometry may reject every pair | Classify as fixed-identity tuning result only |
| Epsilon 0.985385 warm start | Prior PP-UKF bootstrap diagnostic | May be poor for large L | Independent dual averaging at every L |
| 64 adaptation steps | Reviewed bounded broad-grid precedent | Epsilon may not stabilize | Reject nonfinite/non-common endpoint; record as repair trigger |
| 128-result screens, three replications | Operational broad policy requires evidence beyond a 64-draw nomination | Still limited and initial-state dependent | Explicit heuristic/non-convergence status |
| Serial one-process GPU | One physical GPU and graph reuse | Long wall time | L=3 canary and prospective 50%-margin gate |
| Four-GPU-hour ceiling | Existing PP-UKF continuation boundary | May be under-budgeted | Stop rather than weaken grid or screens |

## Pre-Mortem

- A successful command could still tune the wrong target. Bind target,
  transport, adapter, coordinate, metric, start-bank, and source signatures.
- A primary could appear tuned by inheriting another L's epsilon. Record the
  tune seed and tune-evidence signature separately for every primary.
- Neighbor retuning could hide local L sensitivity. The guard callback accepts
  only the inherited epsilon and never constructs an adaptation kernel.
- Cheap acceptance alone could hide stuck or invalid chains. Require finite
  samples/traces, all-chain movement, target status, and native divergence
  telemetry when exposed.
- A partial grid could be mistaken for a result. The complete scheduler barrier
  must pass before guards or viability interpretation.

## Skeptical Pre-Execution Audit

- Wrong baseline: repaired. The comparison is against the same PP-UKF target's
  failed anchor/bootstrap route, not another model or metric.
- Proxy promotion: acceptance means and timing do not rank candidates; pair
  viability also requires hard-veto checks and complete barriers.
- Missing stop: the L=3 resource canary, four-hour ceiling, memory policy,
  identity/lineage checks, and complete barriers are explicit.
- Unfair comparison: every L uses the same target, transport, identity metric,
  starts, adaptation length, screen length, and replication count; only L and
  its independently tuned epsilon vary.
- Hidden assumptions: warm-start epsilon, adaptation length, initial bank,
  screen power, and fixed identity are classified above.
- Stale context: the active reviewed primary grid is used directly; the old
  anchor-local Phase-5 grid is not reused.
- Environment mismatch: serious GPU commands require trusted GPU access,
  XLA, and verified memory growth before device initialization.
- Artifact insufficiency: the driver must write a run manifest, canary/resource
  decision, private pair payload, public summary, timings, device policy, and a
  terminal result note.

Audit decision: `PASS_FOR_BOUNDED_EXECUTION`. The plan can establish broad-grid
pair viability and local same-epsilon suitability. It cannot establish a best
kernel, convergence, posterior validity, or default readiness.

## Attempt 01 Audit Amendment

Attempt 01 completed the real `L=3` primary and stopped at its resource gate.
The gate projected every possible neighbor of all six primaries before the
primary barrier had established which primaries were viable. This was safe but
materially over-conservative: guard existence is conditional on primary
viability, and the observed `L=3` primary was not viable. Charging hypothetical
guards before the primary barrier therefore did not answer whether the required
six-primary search fit the campaign budget.

The corrected staged gate above preserves the same target, method, screens,
hardware, four-hour ceiling, and stop rules. Attempt 01 is preserved and its
wall time is charged. Attempt 02 may run the complete primary barrier only if
the primary-only projection fits, then must separately gate the exact guard set.
This is a localized resource-harness repair under the unchanged scientific
contract, not a budget or direction change.
