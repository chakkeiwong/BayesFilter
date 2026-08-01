# P4 Repair Record

Date: 2026-07-15

## PP-UKF PF Target Admission Attempt 01

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/pf-target-admission/attempt-01-20260715T115041Z/`

Classification: `INFRASTRUCTURE_PF_ORCHESTRATION_TIMEOUT`.

The trusted GPU/XLA run remained compute-bound with bounded memory but crossed
the predeclared 20-minute PF wall-time ceiling. No rung artifact, stabilized PF
reference, filter decision, identity, HMC, or training result was produced.

Root cause: `tf.random.stateless_categorical` was asked to draw N samples from
N logits for every seed/time at PF1 N=16,384. That kernel has pathological work
for this use even though multinomial resampling itself does not require it.

Repair: preserve exact multinomial resampling and every scientific boundary,
but implement it by drawing N independent stateless uniforms and applying the
inverse categorical CDF with batched `tf.searchsorted`. Particles, seeds,
audit points, PF rungs, time order, criteria, hardware class, and total budget
remain unchanged. Run focused deterministic/XLA tests and a PF1-scale timing
probe before retrying in a fresh root.

## PP-UKF PF Target Admission Attempt 02

Output root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/pf-target-admission/attempt-02-20260715T121448Z/`

Classification: `INFRASTRUCTURE_RECOMPOSITION_CALLABLE_BINDING`.

The repaired PF computation and filter screen reached the typed-identity branch,
which means their in-process gates passed. Identity issuance then failed because
the runner passed a callable recomposer instance to the source-inspection API;
that API requires the inspectable bound `__call__` method. The exception occurred
before `result.json` and raw rung rows were persisted, so no PF or filter result
is claimed or reconstructed from this attempt.

Repair: create the recomposer instance and pass
`likelihood_recomposer.__call__`, matching the admitted P2/P3 pattern. Add a
focused independent-recomposition regression, then rerun the unchanged ladder
in a fresh attempt root.
