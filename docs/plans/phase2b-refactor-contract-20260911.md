# Phase 2B Binding Refactor Contract

**Date:** 2026-09-11  
**Branch:** `surrogate-hmc`  
**Parent:** [LEDH Surrogate-Force HMC Unified Program](ledh-surrogate-hmc-unified-program-2026-09-06.md)  
**Status:** BINDING FOR PHASE 2B IMPLEMENTATION

## Objective

Replace the duplicated single-cloud and fused-batch LEDH finite programs with
one semantic engine. Public functions remain adapters; algorithm stages,
configuration branching, analytical tangents, and diagnostics live once.

## Existing public contracts

- `canonical_value_and_analytical_score`: one parameter vector `[P]`; value is
  scalar and score is one analytical directional derivative `[1]` under the
  model callback's closed-over direction convention.
- `canonical_batch_value_score`: reference adapter over theta rows `[B,P]`;
  currently limited to one parameter direction and not training-eligible.
- `canonical_batch_fused_value_score`: theta `[B,P]`, directions `[B,K,P]` (or
  `[B,P]` compatibility form), value `[B]`, directional scores `[B,K]`.
- `CanonicalNeuTraTarget.batch_value_score`: delegates to the fused entry point.

The refactor must not reinterpret `[P]`, `[B,P]`, or `[B,K,P]`. In particular,
`K` is a direction axis only in the fused API; it is not a theta batch in the
single-cloud API.

## Configuration hierarchy

Supported behavior is preserved as a hierarchy, not an unrestricted Cartesian
product:

1. `reset_policy="none"`: historical diagnostic route. Reset and all nested
   correction controls have no effect. It is never production-compliant.
2. `reset_policy="contract_e"`: canonical reset route.
   - `correction_steps == 0`: diagonal correction and trust controls inactive.
   - `correction_steps > 0`: diagonal correction active; trust radius zero and
     nonzero are distinct supported variants.
   - `pairwise_steps == 0`: pairwise strength/RMS cap inactive.
   - `pairwise_steps > 0`: pairwise correction active (structural no-op at d=1).
   - `coordinate_cap == 0`: standardized cap inactive.
   - `coordinate_cap > 0`: standardized cap active after correction.
3. `annealed_stages == 1` and `>1` are supported for reset-none and Contract-E,
   subject to existing trace/callback composition restrictions.
4. `with_score=False` must execute the identical value program and return no
   score; it must not select a different numerical route.

Off variants are preserved diagnostics/ablations. Production remains pinned to
Contract-E with dual-cap and trust-region enabled by
`LEDH_PRODUCTION_PROGRAM_V1`; an off variant must fail if labeled production.

## Non-regression requirements

1. The parameter-sensitive baselines in
   `tests/highdim/test_ledh_configuration_regression.py` remain within rtol
   `1e-12` in float64 for value and score.
2. Inactive controls remain exact no-ops, as tested there.
3. Active adjacent configurations remain observably distinct on the fixture.
4. Fused B/K semantics pass exact row-isolation for B in `{1,2,4}` and K in
   `{1,2,5}`.
5. Full-union row isolation is added before the old engines are retired: every
   valid configuration family runs at B in `{1,2,4}` and representative K.
6. Same-lane pre/post parity uses rtol `1e-12` (float64). Cross-lane parity may
   use `5e-4` only for unavoidable operation-order differences.
7. Hand-derived JVPs continue to pass the existing autodiff-oracle tests; the
   oracle never enters a claim-bearing path.
8. Diagnostics are preserved or extended. Missing fields, changed meanings, or
   row aggregation are vetoes.

## Implementation constraints

- One stage implementation for single and batch adapters.
- Pointwise work may flatten `[B,N,...]` to `[B*N,...]`; every cloud reduction
  must retain an explicit B axis or equivalent segment ID.
- No Python loop over theta rows or directions in a traced training graph.
  TensorFlow control flow over fixed algorithm iterations is permitted.
- No `tf.vectorized_map`/pfor and no autodiff in the production score path.
- Preserve model callback capability union, covariance tangents, nonlinear
  density callbacks, reset transport, and trace restrictions.
- Preserve repository chunk policy and tuning-scope identities.

## Migration order

1. Introduce batched stage primitives and compare B=1 against single-cloud.
2. Add Contract-E reset and covariance carry; run configuration baselines.
3. Add diagonal, trust, pairwise, and coordinate-cap correction in that order.
4. Add annealed composition and diagnostic payload.
5. Migrate `canonical_value_and_analytical_score` as a B=1 adapter.
6. Migrate fused batch and NeuTra adapters; migrate the reference batch adapter.
7. Run parallel old/new comparisons and full-union row isolation.
8. Delete duplicated stage logic only after all gates pass.

## Promotion vetoes

- Any configuration baseline or inactive-control identity changes.
- Cross-row or cross-direction mixing.
- An off configuration can be labeled production.
- Any analytical JVP oracle failure.
- Reduced diagnostics or changed public shapes/signatures.
- Python row/direction loop, pfor, or claim-path autodiff.
- Retuning required because a bound numerical control changed.

The legacy 327-file LGSSM fixture campaign is supplemental only: it is
incomplete (33/360 missing), its transition ignores theta, and its `K` label was
not the single-cloud API's direction contract. It cannot satisfy this contract.
