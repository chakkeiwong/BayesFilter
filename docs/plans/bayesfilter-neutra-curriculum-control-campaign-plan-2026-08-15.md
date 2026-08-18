# NeuTra curriculum-search Gaussian/banana control campaign plan (2026-08-15)

## Research intent ledger

| Field | Predeclared statement |
|---|---|
| Main question | Can the reviewed curriculum-search procedure choose target-dependent dense-IAF warm-up groups that produce a viable fresh NeuTra proposal on correlated Gaussian and banana controls? |
| Candidate mechanism | Replicated common-parent group probes, bounded beam nomination, an equal-work full-protocol/LR tournament, then a frozen fresh run with untouched exact-law validation. |
| Baseline | Cold joint training is included at every tournament learning rate and receives the identical 3,000-update work per candidate. |
| Primary search criterion | Positive lower uncertainty bound for held-out reverse-KL improvement per probe update within a common parent. |
| Protocol-selection criterion | Paired terminal held-out loss uncertainty set across equal-work `(sequence, LR)` candidates; choose the shortest sequence inside the set. |
| Primary final criterion | Untouched 131,072-draw exact-law mean, second-moment, and adjacent cross-moment screens at 99.9%, separately by target and seed. |
| Probe vetoes | Nonfinite update/loss/state, unequal 100-update probe, changed parent state or incoming loss across sibling candidates, CPU/scalar fallback, invalid group masks, or search-budget violation. |
| Tournament vetoes | Unequal 3,000-update work, unpaired initialization/training/selection partitions, nonfinite run, missing cold arms, or changed architecture. |
| Final vetoes | Any exact-law screen failure, nonfinite artifact, invalid fresh partition, or missing GPU/XLA/memory-growth provenance. |
| Explanatory diagnostics | Probe LCBs, beam paths, tournament paired bounds, selected LR, loss, ESS fraction, ratio SD, moments, clipping, wall time. |
| Must not conclude | A control pass is not SSL-LSTM readiness, universal curriculum validity, HMC correctness, multimodal coverage, or a default training policy. |

## Target and transport scope

Targets are the previously checked exact-law controls:

- correlated Gaussian, dimension 16; and
- banana pushforward, dimension 16.

The transport is the existing three-block width-`(32,32)` ELU dense IAF with
the same initialization family and stage caps used in the prior campaign. The
adapter declares five disjoint groups:

1. `affine_location`;
2. `simple_linear_scale`;
3. `stage_0_residual`;
4. `stage_1`; and
5. `stage_2`.

No prerequisite edge is imposed because the implementation already contains
all blocks and each disjoint group can be activated while other blocks remain
at their identity initialization. This is a mechanically reviewed search
choice, not a claim that every order is scientifically useful.

## Partitions and common random numbers

Each target has distinct deterministic seed namespaces for:

- search initialization and probe-training batches;
- search held-out selection batches;
- tournament initialization and training batches;
- tournament held-out selection batches;
- fresh-final initialization and training batches; and
- untouched exact-law audit draws.

At a given search node and replicate, every sibling candidate uses the same
parent tensor state, incoming held-out loss, training batches, and LR grid.
Every tournament candidate uses the same initialization, global-step training
batches, and selection partition for its replicate. Audit draws never enter
search, LR selection, or the tournament.

## Search configuration

| Setting | Value | Provenance and limitation |
|---|---:|---|
| Probe updates | 100 | Inherited low-dimensional diagnostic budget; a short nomination probe, not convergence evidence. |
| Probe LR grid | `2e-4`, `5e-4`, `1e-3` | Existing target campaign grid; every sibling gets the same grid. |
| Probe replicates | 4 | Minimum planned replication with a dispersion estimate. |
| Beam width | 2 | Bounded combinatorial search hypothesis. |
| Maximum depth | 3 | At most three warm-up groups before joint training. |
| Maximum probe calls | 80 | Covers the worst planned width/depth expansion without partial replication. |
| Critical value | 2.0 | Predeclared conservative one-sided uncertainty multiplier. |
| Minimum improvement/update | 0 | Derived null: a group must have a positive lower uncertainty bound. |
| Batch size | 4,096 | Existing batch-native GPU/XLA baseline. |
| Probe selection rows | 65,536 | Existing held-out size; selection only. |

Before search, measure the held-out loss of the exact known-law transport on 16
independent 65,536-row calibration batches. Let `s_exact` be the sample SD of
those batch means and define the independently measured repeatability margin
`m = 2 * s_exact`. The search uses
`minimum_improvement_per_update = m / 100`. Thus a group must improve faster
than the measured two-SD loss-repeatability scale over one probe. The exact-law
calibration batches are disjoint from all trained candidates and audits.

Each probe LR candidate consumes exactly 100 updates from the identical parent.
The terminal held-out loss selects the LR and child state. Actual probe tuning
work is therefore `3 * 100` optimizer updates per replicate/candidate and is
reported separately from the selected path.

## Equal-work full-protocol tournament

The final beam supplies at most two sequences. Add the empty sequence as cold
joint. Cross each sequence with every LR in the same three-point grid. Each
`(sequence, LR)` pair is a distinct tournament candidate and consumes exactly
3,000 optimizer updates:

- each warm-up group in sequence receives the first 100 global updates in its
  cumulative prefix;
- all remaining updates train every transport variable jointly;
- the LR follows the same global piecewise schedule for every candidate:
  multiplier `1` before 60%, `0.1` from 60%-85%, and `0.01` afterward; and
- Adam state resets at warm-up phase boundaries, while cold joint has no phase
  boundary. This reset behavior is part of the candidate protocol and is
  recorded.

There is no phase-local LR tuning in the tournament. Thus every candidate uses
exactly 3,000 optimizer updates per replicate. Four paired replicates run for
each candidate.

The practical loss tolerance is the independently calibrated target-specific
margin `m = 2 * s_exact`. It measures repeatability of the same 65,536-row
held-out loss statistic under the exact transport. It is used only to form the
protocol-selection uncertainty set and cannot relax the final exact-law gate.
The selector chooses the shortest sequence inside that paired uncertainty set;
name order resolves only an exact complexity tie.

## Fresh final confirmation

Freeze the selected sequence and LR. Also freeze the cold LR with the lowest
mean tournament loss as an explicit comparator. Run two fresh seeds per unique
candidate, each with new initialization/training partitions and exactly 3,000
updates. Apply the untouched exact-law audit after training.

The candidate is viable only if both fresh seeds pass every exact-law screen.
A one-seed pass is descriptive and triggers replication, not promotion. If the
searched candidate is itself the cold comparator, run it once per fresh seed
and classify the search as selecting cold joint training for that target.

## Skeptical plan audit

| Risk | Disposition |
|---|---|
| Fixed order hidden in the adapter | Vetoed: five disjoint groups have no imposed order; beam search chooses prefixes. |
| Local probe scores compared across parents | Vetoed: local LCBs only build the beam; tournament makes the cross-branch decision. |
| Staged protocols get more LR tuning | Vetoed: each `(sequence, LR)` is a separate equal-3,000-update tournament arm. |
| Cold baseline is weak or under-tuned | Vetoed: all three cold LR arms enter the paired tournament. |
| Protocol selection uses final audit | Vetoed: tournament selection and audit seed namespaces are disjoint. |
| A short probe rejects a delayed-benefit group | Retained risk: beam width two and cold baseline limit damage; result cannot reject the scientific direction. |
| Probe threshold is arbitrary | Repaired: use exact-transport loss repeatability divided by the probe budget. |
| Protocol tolerance becomes a correctness margin | Vetoed: independently measured exact-transport repeatability affects nomination only; exact-law screens remain unchanged. |
| Seed pairing is invalid | Vetoed: candidates share initialization, global-step batches, and selection partition by replicate. |
| Search passes while final training fails | Expected possible outcome; classify as protocol instability and reject the candidate. |
| Campaign exceeds compute | Hard one-hour wall cap and 80-probe-call cap. Preserve progress and stop. |

Audit verdict: the plan gives every tournament candidate equal optimizer work,
keeps proxy loss separate from exact-law validity, and can answer whether the
search procedure selects a viable target-specific protocol on these controls.

## Execution and artifacts

- TensorFlow/TFP GPU route, float64, XLA JIT, TF32 disabled.
- `TF_FORCE_GPU_ALLOW_GROWTH=true` and repository growth verification before
  logical-device initialization.
- One target process at a time on GPU 0 to avoid adding cross-agent contention.
- Campaign wall cap: 3,600 seconds.
- Output root:
  `docs/plans/artifacts/neutra-curriculum-control-campaign-2026-08-15/`.

Artifacts must include the exact commands, git commit, environment, device,
memory policy, group definitions, every probe observation/candidate, beam,
tournament observations/selection, fresh-final manifests/results, wall times,
state hashes, and SHA-256 integrity manifests.
