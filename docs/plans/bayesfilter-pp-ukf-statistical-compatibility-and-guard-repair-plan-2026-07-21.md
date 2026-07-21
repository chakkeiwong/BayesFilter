# PP-UKF Statistical Compatibility And Guard Repair Plan

Date: 2026-07-21

Status: `COMPLETE_VIABLE_PAIR_SET`

Terminal result: the corrected classifier retained primaries
`L=(5,9,13,18,25)` as statistically compatible and rejected only `L=3`.
Fresh exact-epsilon guards established locally suitable neighborhoods for
`L=(13,18,25)`. All nine guards completed; cumulative charged time was
`13,750.560450 s`, below the unchanged `14,400 s` ceiling. See
`docs/plans/bayesfilter-pp-ukf-statistical-compatibility-and-guard-repair-result-2026-07-21.md`.

## Objective

Correct the PP-UKF tuning classifier so it rejects an epsilon only when fresh
replicated evidence is statistically separated from the practical acceptance
band `[0.65,0.75]`. Preserve the twelve chain-level diagnostics, but use the
three independently seeded replication means as the uncertainty units. Apply
the corrected rule retrospectively to the completed six-primary artifact, then
run the newly required exact-epsilon one-hop guards under the unchanged
`14,400 s` campaign ceiling.

The prior state-continuing artifact remains immutable. Its final-screen values
are eligible for reclassification because the statistical rule changes but the
values, seeds, target, metric, transport, and state lineage do not. Its previous
`inconclusive_evidence` disposition is superseded, not overwritten.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Which PP-UKF primary epsilon arms are statistically compatible with mean acceptance in `[0.65,0.75]`, after accounting for replication noise? |
| Candidate | Each completed `(L, epsilon_L)` primary, followed by exact-epsilon one-hop guards for compatible primaries |
| Claimed target | Mean Metropolis acceptance probability in `[0.65,0.75]`, centered nominally at `0.70` |
| Statistical unit | Mean across four chains within each of three independently seeded final replications |
| Working interval | Two-sided 90% Student-t interval over three replication means, `df=2`, critical value `2.919985580355516` |
| Rejection criterion | `needs_lower_epsilon` only when interval upper bound is below `0.65`; `needs_higher_epsilon` only when interval lower bound is above `0.75` |
| Nomination criterion | Interval overlaps `[0.65,0.75]`, with no hard veto; label `provisional_viable` |
| Promotion veto | Existing hard target, state, movement, identity, lineage, artifact, or resource veto; incomplete primary or required-guard barrier |
| Continuation veto | Cumulative charged wall time or staged projection exceeds `14,400 s`; reconstructed calibrated state or epsilon differs from the preserved primary artifact |
| Explanatory only | Point mean, chain-level values, interval width, calibration path, and runtime |
| Must not be concluded | Interval overlap does not prove the true acceptance is in-band, rank candidates, establish convergence, or authorize retained sampling |

## Evidence And Execution Contract

- Source artifact:
  `docs/plans/artifacts/bayesfilter-pp-ukf-state-continuing-epsilon-repair-20260721/attempt-01/private_result.json`.
- Target signature:
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Reclassification must reproduce every stored point mean from the twelve
  chain means and preserve the source evidence signature as lineage.
- Parent calibration is reconstructed only for compatible primaries. Exact
  tuned epsilon and calibrated-state SHA-256 must match the source artifact.
- Guards use the parent's epsilon bit-for-bit, do not retune, start from the
  reconstructed matching parent state, and use three fresh 96-result screens
  after eight initialization transitions.
- Every guard uses the same replication-mean 90% interval and overlap rule.
- No posterior or retained sampling is authorized.
- Fresh artifact root:
  `docs/plans/artifacts/bayesfilter-pp-ukf-statistical-compatibility-guard-repair-20260721/`.

## Resource Contract

The unchanged campaign ceiling is `14,400 s`. Charged before this repair is
`9,931.762329853023 s`; remaining headroom is `4,468.237670146977 s`.

Use measured source times to project only the work made necessary by the
corrected rule:

- reconstruct calibration for statistically compatible parents only;
- run the actual deduplicated exact-epsilon guard set;
- separate one-time runner compilation from steady per-leapfrog execution;
- apply a 25% margin to both components; and
- stop before launch if projected cumulative time exceeds the ceiling.

The source observations imply compatible primaries `L=(5,9,13,18,25)` and
guards `L=(4,6,8,10,12,14,17,19,24)` at their parent-specific epsilons. The
measured reconstruction time is about `724 s`; measured steady guard work is
about `2,070 s`. With one-time compilation and 25% margin, the projection is
below the remaining `4,468 s`. The executable must recompute and record the
projection rather than trusting these rounded values.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Reject only on statistical separation | User directive; reviewed tuning-nomination rule | Very noisy arms remain candidates | Label overlap as compatibility, not proof; require guards and later validation |
| Replication mean as unit | Three disjoint final-screen seeds; statistical correction | Only three units give wide, assumption-sensitive intervals | Preserve raw chain means and report `df=2` limitation |
| 90% interval | Existing broad-grid heuristic level | More permissive than 95% | Report level; note that current `L=3` verdict is unchanged at 95% |
| Student-t approximation | Small-sample working model | Non-Gaussian replication means | Heuristic-only label; no ranking or convergence claim |
| Retrospective reclassification | Same unchanged numerical evidence | Post-hoc rule could overstate evidence | User supplied the rule; preserve source artifact and issue a versioned correction |
| Reconstructed parent states | Deterministic same code, seeds, target, and GPU route | Drift would make guards incomparable | Exact epsilon and state-signature equality are continuation vetoes |

## Pre-Mortem

- The code could retain the overlap rule but still use twelve correlated chain
  means. Test the exact three replication groups and `df=2` critical value.
- A point estimate just outside the band could still be rejected despite a
  wide overlapping interval. Add upper- and lower-side noisy regression cases.
- Compatibility could be described as proof of in-band acceptance. Bind the
  nonclaim in code, artifacts, and result text.
- Reconstructed calibration could drift silently. Fail before guards unless
  both epsilon bits and calibrated-state signature match.
- Guard expansion could merge equal `L` values across distinct parent epsilons.
  Preserve `(guard L, epsilon bits)` as the identity.
- Replaying all primaries could waste the budget. Reconstruct only compatible
  parents and gate the actual guard set before launch.

## Skeptical Pre-Execution Audit

- **Wrong baseline:** repaired. The numerical evidence is unchanged; only its
  explicitly user-corrected statistical interpretation changes.
- **Proxy promotion:** interval overlap nominates a tuning candidate only.
  Guards remain mandatory and no sampling or convergence claim follows.
- **Missing stop:** exact reconstruction equality, primary/guard barriers,
  staged projection, and the unchanged campaign ceiling are hard stops.
- **Unfair comparison:** every primary and guard uses the same replication
  aggregation, interval level, and overlap decision.
- **Hidden assumptions:** small-sample t approximation, interval level,
  dependence structure, and retrospective nature are explicit.
- **Stale context:** source artifact hashes, target, transport, epsilon, state
  signatures, and remaining charged budget are bound.
- **Environment mismatch:** fresh guards require the reviewed GPU/XLA float64
  path and verified memory growth.
- **Artifact insufficiency:** corrected primary rows, reconstruction receipts,
  every guard screen, barrier decisions, resource projection, and manifest are
  required.

Audit decision: `PASS_FOR_BOUNDED_EXECUTION`. The earlier strict-containment
rule answered the wrong tuning question, and treating twelve within-replication
chain means as independent understated uncertainty. The corrected rule answers
statistical compatibility, preserves nonclaims, and fits the unchanged budget.

## Planned Changes And Checks

- Correct the shared classifier and expose replication-level interval fields.
- Add focused statistical-unit, overlap, and separation tests.
- Add a versioned retrospective reclassifier and bounded guard-repair driver.
- Run CPU-hidden focused and adjacent tests before GPU execution.
- Run trusted GPU preflight, exact parent reconstruction, and fresh guards.
- Verify terminal artifacts and write a superseding result note.
