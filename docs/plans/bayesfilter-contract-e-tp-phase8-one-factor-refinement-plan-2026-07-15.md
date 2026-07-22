# Contract E--TP Phase 8 One-Factor Refinement Plan

metadata_date: 2026-07-15
status: READY_AFTER_SKEPTICAL_AUDIT
program_id: contract-e-tp-all-model-gradient-comparison
phase: 8
execution_target: explicit CPU-hidden TensorFlow float64 diagnostics
budget: 16 CPU core-hours; at most three attempts per rung

## Phase Objective

Determine whether the scalar adjacent-state extension's Phase 7 error is
primarily polynomial degree, quadrature, TT rank, coordinate support, or sweep
capacity, using the smallest failing `T=1` prefix and changing one factor at a
time.  Separately preserve generalized Contract E--TP as a negative result for
the already-tested progressive continuation feature span; do not repeat anchor
count changes that leave that span unchanged.

## Entry Conditions

- Phase 7 controlling ledger v2 and result exist.
- Every populated TP/extension score passes own-scalar FD.
- Scalar TP/extension observation-prefix hashes match exactly.
- The extension is classified `extension_or_invention`, never Zhao--Cui.
- Generalized TP `T=2` fails the dense reference in `gamma` despite exact
  chart feature matching and passing FD.

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Can one-factor capacity changes materially reduce the extension's `T=1` value/score error without invalidating its finite program? |
| Baseline | degree 8, order 17, rank 2, two sweeps, coordinate half-width 8 |
| Primary diagnostic | maximum componentwise score error to the same dense reference |
| Hard veto | target hash/time-order mismatch, FD failure, nonfinite, invalid fit, mass failure, or condition veto |
| Explanatory diagnostics | value error, fit residual, scaled/unscaled condition, runtime |
| Promotion criterion | none; rungs nominate the next factor only and do not establish defaults or equivalence |
| Continuation rule | a valid but inaccurate rung continues to the next one-factor diagnostic; only invalid evidence stops |
| Artifact root | `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8_refinement_20260715/` |

All extension runs must consume embedded `T=1` target preparations.  The
repaired actual-SV preparation is already embedded.  Create embedded KSC and
generalized-SV preparations before the ladder and require their serialized
target-observation hashes to equal the corresponding Phase 7 controlling
hashes. No seed-only regeneration is admissible.

The frozen preparation paths are:

- `phase7_actual_sv_t1_bound_target_preparation_20260715.json`;
- `phase8_ksc_sv_t1_bound_target_preparation_20260715.json`;
- `phase8_generalized_sv_t1_bound_target_preparation_20260715.json`.

All paths are relative to
`docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/`.

## One-Factor Ladder

Run actual, KSC, and generalized SV at `T=1` in this order:

1. degree ladder: 12 and 16, holding order 17, rank 2, sweeps 2, width 8;
2. quadrature ladder: order 25 and 33, holding degree 8, rank 2, sweeps 2,
   width 8;
3. generalized only, rank ladder: rank 3 and 4, holding degree 8, order 17,
   sweeps 2, width 8;
4. only after inspecting steps 1--3, choose the single most likely next factor:
   coordinate width or fixed sweep count.  Do not run both without revising the
   result/next-step record.

Interim evidence amendment: the initial arms show that quadrature alone is
flat, actual SV improves materially with degree, and generalized SV improves
monotonically in both value and score from rank 2 to 3 to 4. KSC degree/order
changes trade score improvement against worse value. Continue the demonstrated
generalized rank family at ranks 6 and 8. For the one-dimensional actual/KSC
fits, test coordinate half-widths 6 and 10; extra sweeps are not selected
because `T=1` is the same fixed linear least-squares target and no
iteration-dependent state is being carried. These are still one-factor arms.

Second interim amendment: width 6 improves both value and score for actual and
KSC, while width 10 worsens both. Actual degree 12 also improves both and is
better than width 6 on the `T=1` descriptive gaps. Generalized rank improves
through rank 4 but ranks 6/8 overshoot the small `gamma` score and worsen score
error even while value and fit residual continue improving. Therefore:

- extend actual degree 12, order 17, rank 2, width 8 to `T=2`;
- extend KSC degree 8, order 17, rank 2, width 6 to `T=2`;
- do not advance generalized rank 4 because its `T=1` score error remains large
  and the rank sequence is nonmonotone in the primary score diagnostic.

The two advancing configurations are hypotheses, not selected defaults. Their
`T=2` runs must use fresh embedded preparations whose target hashes are checked
against the Phase 7 `T=2` controlling cells. A `T=2` failure is candidate
rejection and does not stop Phase 8 or invalidate the harness.

Each rung artifact name must encode row, varied factor/value, and
`phase8_refinement`; prior evidence is never overwritten. The controlling
aggregation artifact is
`phase8_refinement_20260715/refinement_ledger.json`.

At `T=1`, rank is inapplicable to actual/KSC because their first step is
one-dimensional rank `(1,1)`.  Generalized uses the two-axis transitioned
initial step, so rank is meaningful.

Do not run longer horizons until a `T=1` capacity family shows stable movement
toward the reference.  A lower fit residual without lower value/score error is
not a promotion.

## Generalized Contract E--TP Boundary

The existing progressive continuation basis `(1,4,9)` with 8/12 quantile
anchors has already shown nonmonotone refinement and a `T=2` sign reversal.
More anchors in the same span are forbidden as a claimed repair.  A future
repair must define and justify a materially different distributional summary,
for example a fixed orthogonal basis of the continuation value function with a
truncation/residual bound.  Designing that feature family is a scientific
subphase; it is not silently included in this capacity ladder.

## Required Checks

1. exact target-preparation path and hash in each artifact;
2. realized first-step axis/time-order evidence;
3. own-scalar FD, marginal mass, finite value/score, and fit status;
4. table against the same Phase 7 dense reference;
5. one-factor diff audit proving all non-varied fields are equal;
6. focused tests and `git diff --check`;
7. result note with decision/inference tables and post-run red team.

## Skeptical Plan Audit

Status: `PASS_AFTER_PROXY_AND_CAPACITY_REPAIR`.

The tempting plan to run longer horizons or increase degree, order, rank, and
sweeps together would not identify a cause.  The repaired plan starts at
`T=1`, changes one factor, preserves exact data bytes, and treats fit residual
as explanatory rather than a promotion proxy.  It does not reopen the closed
generalized-TP anchor-count family.  There is no arbitrary agreement threshold;
all gaps remain descriptive.

## Handoff And Stop Conditions

If a one-factor family shows stable improvement in score and value, Phase 8
may extend only that nominated configuration to `T=2`, then `T=10`.  If no
family improves, close the fixed degree/order/rank hypothesis as a negative
result and identify coordinate support or optimization sweeps as the next
smallest diagnostic.  GPU/XLA Phase 9 begins only for correctness-eligible TP
rows, not for an inaccurate extension or the generalized negative TP row.
