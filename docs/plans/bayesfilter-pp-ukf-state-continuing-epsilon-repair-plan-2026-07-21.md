# PP-UKF State-Continuing Epsilon Repair Plan

Date: 2026-07-21

Status: `COMPLETE_SUPERSEDED_STATISTICAL_CLASSIFICATION`

Terminal result: the repaired protocol completed all six primaries on the
reviewed GPU/XLA route in `2,937.756936 s`. No primary satisfied strict
replicated admission, so no guard was eligible and no sampling was launched.
Total charged campaign time was `9,931.762330 s`, below the unchanged
`14,400 s` ceiling. See
`docs/plans/bayesfilter-pp-ukf-state-continuing-epsilon-repair-result-2026-07-21.md`.

Statistical correction: the strict interval-containment promotion rule in this
historical plan was too conservative for tuning nomination and used correlated
chain means as if they were independent units. The numerical run is preserved,
but its classification is superseded by
`docs/plans/bayesfilter-pp-ukf-statistical-compatibility-and-guard-repair-plan-2026-07-21.md`.

## Objective

Repair the invalid one-pass PP-UKF broad-grid tuning protocol without raising
the existing four-GPU-hour campaign ceiling. Independently tune epsilon for
`L=(3,5,9,13,18,25)` while carrying the chain state from adaptation into
calibration, repair out-of-band epsilon with bounded fixed-kernel calibration
steps, admit a primary only when its fresh replicated working interval is fully
inside `[0.65,0.75]`, and only then run exact-epsilon one-hop neighbor guards.

The previous `viable_pair_set` result is withdrawn. Its epsilon values are
warm starts and cost observations only. Its samples, acceptance evidence, and
candidate disposition are not reused as fresh evidence.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Can state-continuing adaptation plus bounded epsilon repair tune each primary L into the declared acceptance region under the remaining campaign budget? |
| Candidate | One independently repaired `(L, epsilon_L)` pair per primary L, followed by exact-epsilon one-hop guards for strictly viable primaries |
| Claimed target | Mean Metropolis acceptance probability 0.70; admissible practical region `[0.65,0.75]` |
| Quantity computed | `exp(min(log_accept_ratio, 0))`, averaged within fresh chain runs and then summarized across 12 chain-run means |
| Expected failure mode | The 96-step adaptation endpoint can still be biased; short calibration can be noisy; a final interval can cross a band boundary even when its mean lies inside |
| Promotion criterion | Point mean inside `[0.65,0.75]`, complete working interval inside `[0.65,0.75]`, no hard veto, complete six-primary barrier, and complete required guard barrier |
| Promotion veto | Target/transport/metric/state-lineage drift; nonfinite state/value/score/log acceptance; invalid target status; failed movement/path-return; exposed positive divergence; incomplete barrier |
| Continuation veto | Cumulative charged wall time or a conservative staged projection exceeds 14,400 s; GPU memory policy fails; required evidence is unavailable or corrupted |
| Repair trigger | Calibration mean above 0.72 increases epsilon; below 0.68 decreases epsilon; bracketed repairs use a geometric midpoint |
| Explanatory only | Adaptation history, calibration means, epsilon changes, timing, allocator use, and out-of-band rejected candidates |
| Must not be concluded | No posterior convergence, retained-sampling readiness, stochastic superiority, target correctness, or default readiness follows |

## Evidence Contract

- Target signature:
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Fixed identity metric and unchanged four-chain start bank.
- Backend: TensorFlow/TFP GPU/XLA, float64, verified GPU memory growth.
- Prior epsilon warm starts, by L:
  `(3:0.8724049589170738, 5:0.8426345584765329,
  9:0.7489709357241571, 13:0.69086551957137,
  18:0.6813265222611998, 25:0.6800917535732008)`.
- Adaptation: 96 dual-averaging transitions targeting 0.70, immediately
  followed in the same TFP call by 32 frozen-epsilon transitions. The returned
  final state becomes the calibration state. The 32-transition acceptance mean
  is calibration evidence only.
- Calibration repair: at most three additional state-continuing fixed-kernel
  calls, each with one initialization transition and 32 calibration results.
  Above 0.72 sets/updates a lower epsilon bound; below 0.68 sets/updates an
  upper bound. An unbracketed proposal changes epsilon by factor 1.20; a
  bracketed proposal is the geometric midpoint. Calibration stops inside
  `[0.68,0.72]`.
- Final primary evidence: three fresh replications, each 96 results after eight
  initialization transitions, all starting from the frozen calibrated state
  and using disjoint seeds. Calibration draws are excluded.
- Classification: a point mean outside `[0.65,0.75]` receives a directional
  repair disposition. A point mean inside while its interval crosses either
  boundary is `unresolved_budget`, never viable. Only a fully contained
  interval is `provisional_viable`.
- Guards: one hop, nonrecursive, exact parent epsilon, no retuning; each guard
  begins from its parent calibrated state and uses the same three-by-96 screen.
- Artifact root:
  `docs/plans/artifacts/bayesfilter-pp-ukf-state-continuing-epsilon-repair-20260721/`.

## Resource Contract

The campaign ceiling remains `14,400 s`. Charged before this repair:

- earlier PP-UKF operational work: `3,930.3757156120264 s`;
- completed broad-grid Attempt 02: `3,063.6296786410094 s`;
- total charged: `6,994.005394253036 s`;
- remaining: `7,405.994605746964 s`.

Before launch, project one adaptation/calibration/final-screen pass for all six
primaries using the measured per-L tune and screen rates from the terminal
broad-grid artifact, the maximum three repairs, and a 50% margin. Stop if that
projection exceeds the remaining budget. During execution, preserve progress
after every primary. After the primary barrier, project only the actual guard
set using current measured rates and the same margin. Never weaken evidence
lengths after observing a result to make the budget fit.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Target 0.70 | User-confirmed and passed correctly to TFP in prior run | Acceptance statistic mismatch | Bind and record TFP target plus screen statistic identity |
| 96 adaptation + 32 frozen transitions | Repair hypothesis; longer and structurally correct relative to prior 64+restart | Still insufficient stabilization | Constant post-adaptation epsilon and calibration mean |
| State continuation | Direct repair of observed mismatch | Final state could be invalid | Combined value/score/status and finiteness check after every call |
| Calibration target `[0.68,0.72]` | Centered within promotion band | Noise may stop too early | Fresh three-replication final screen is authoritative |
| Factor 1.20 and geometric bracketing | Bounded scalar root-search hypothesis | Overshoot or slow bracketing | Three-repair cap, explicit bounds, fresh means |
| Three repairs | Budgeted convenience cap | Some L remains untuned | Preserve directional failure; do not call it viable |
| Three-by-96 final evidence | More than 64-draw nomination and within remaining budget | Lower precision than prior 128 screens | Strict interval containment; no ranking claim |
| Prior epsilons | Same-target diagnostic warm starts | Prior invalid result biases search | No prior state/evidence reuse; fresh adaptation at every L |
| Fixed identity | Existing campaign scope | Poor geometry | Scope-limited result and no default claim |

## Pre-Mortem

- The command could report success while still using the original states for
  the final screens. Bind each screen to the calibrated-state content signature
  and test that the continuation state is passed explicitly.
- Epsilon could be adjusted in the wrong direction. Unit-test high acceptance
  -> larger epsilon and low acceptance -> smaller epsilon, including bracketed
  midpoints.
- An overlap rule could again admit an out-of-band mean. Add boundary tests for
  point means and intervals separately.
- The extra 32 results could still be adapting due to an off-by-one error.
  Require the entire returned step-size trace to be constant and equal to the
  endpoint epsilon.
- Calibration could overfit its own draws. Final screens use disjoint seeds and
  exclude calibration evidence.
- Guard cost could silently exceed the campaign. Gate the actual expanded guard
  set after the primary barrier.

## Skeptical Pre-Execution Audit

- **Wrong baseline:** repaired. The same target, frozen transport, identity
  metric, primary L grid, and start bank are retained; only the invalid tuning
  protocol changes.
- **Proxy promotion:** calibration means can steer epsilon but cannot promote a
  candidate. Promotion uses fresh replicated evidence and strict containment.
- **Missing stop:** total campaign cap, prospective primary gate, progress
  checkpoints, repair cap, and actual-guard gate are explicit.
- **Unfair comparison:** every L receives the same adaptation, calibration,
  final-screen lengths, direction rule, and replication count.
- **Hidden assumptions:** adaptation length, calibration band, factor, repair
  cap, final screen length, and warm starts are recorded above as hypotheses.
- **Stale context:** the plan incorporates the checked TFP endpoint semantics:
  the adaptation endpoint is smoothed correctly, but the previous protocol did
  not continue state or repair its screen.
- **Environment mismatch:** trusted GPU/XLA and verified memory growth remain
  mandatory.
- **Artifact insufficiency:** private progress preserves every epsilon/state
  signature and calibration decision; terminal public/private results,
  resource decisions, and manifest are required.
- **Could pass while misleading:** a tuned acceptance rate does not establish
  convergence or posterior validity; this remains a tuning-only campaign.

Audit decision: `PASS_FOR_BOUNDED_EXECUTION`. The plan directly tests the
diagnosed failure mechanisms and fits the unchanged campaign ceiling under the
prospective measured-rate contract. It cannot establish retained-sampling
readiness.

## Planned Changes And Checks

- Repair strict classification in
  `bayesfilter/inference/hmc_operational_broad_grid.py`.
- Add the state-continuing repair driver under `docs/benchmarks/`.
- Add focused classification, direction, bracketing, budget, state-continuation,
  and artifact tests.
- Mark the previous result as withdrawn rather than deleting historical
  evidence.
- Run CPU-hidden compile and focused/adjacent tests before GPU execution.
- Run trusted GPU preflight, execute one fresh attempt, verify artifacts, and
  write a terminal result note.
