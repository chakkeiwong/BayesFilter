# Zhao-Cui Predator-Prey Fixed-Variant Reset Memo

Date: 2026-07-23  
Current state: `IMPLEMENTATION_GATES_PASS_NOT_ADMITTED`

## What Exists

The predator-prey Zhao-Cui-derived lane now has a source-order fixed APF
evaluator, an offline scope-specific proposal fitter/compiler, focused tests,
and sealed CPU/GPU claim artifacts.

- Evaluator: `bayesfilter/highdim/zhao_cui_predator_prey_fixed_variant_tf.py`
- Proposal fitter: `bayesfilter/highdim/zhao_cui_predator_prey_proposal_tf.py`
- Harness: `docs/benchmarks/run_zhao_cui_predator_prey_fixed_variant.py`
- Source audit: `docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-source-audit-2026-07-23.md`
- Active plan: `docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-active-plan-2026-07-23.md`
- Result: `docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-result-2026-07-23.md`

Authoritative evidence is the pair:

- CPU replay: `docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_cpu_claim_repaired_20260723_1100/result.json`
- GPU claim: `docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_gpu_claim_final_20260723_1145/result.json`

The two artifacts have identical frozen branch/program IDs and hashes. The
earlier CPU claim and GPU claim directories remain historical attempts and must
not be used as the current branch because their preparation device was not
fixed.

## Current Route Contract

- Target: `zhao_cui_predator_prey_tf_seed81104_x0_then_y1_y20_v1`.
- Event order: `x0 -> 20 transitions -> y1:y20`.
- Route ID: `zhao_cui_predator_prey_source_order_fixed_branch_extension_v1`.
- Route classification: `extension_or_invention`.
- Score backend: analytical manual density-score recursion; no runtime FD or autodiff.
- Branch preparation: CPU FP64, frozen before online evaluation.
- Claim execution: GPU FP32, TF32 enabled, XLA enabled, memory growth verified.
- Comparators: CPU-only and descriptive.
- Generic retained-grid Zhao-Cui routes remain historical/diagnostic and are not called.

## Evidence Snapshot

- Value: `-102.41967599576537` (CPU FP64); GPU online `-102.41967010498047`.
- Score `(r,K,a,s,u,v)`: `[-22.67643304, 0.13827965, -0.08341680, 0.24588744, 17.60534898, -22.81586181]`.
- Minimum claim ESS: `448.38/1002`.
- Same-program FD max absolute error: `5.94e-9`.
- GPU allocator current/peak: `437,760 / 10,752,768` bytes.
- Focused tests: `13 passed`.

These are implementation and finite-program consistency results. They are not
evidence of exact likelihood accuracy, posterior correctness, HMC readiness,
statistical superiority, or production scalability.

## Source Boundary

Paper Eq. 13 and Proposition 2 plus the pinned `SIRT`, `marginalise`, and
conditional-evaluation code support the squared-TT and paired-core operations.
Paper Algorithm 3 and `full_sol.m` support sequential proposal correction and
weighting. The frozen source-order finite scalar, finite-grid inverse, and
recursive analytical score are project extensions. Do not use
`source-faithful` for the assembled route.

## Do Not Do

- Do not add the route ID to `ZHAO_CUI_FIXED_VARIANT_ROUTE_IDS` or leaderboard mappings.
- Do not relabel the generic retained-grid evaluator as production Zhao-Cui.
- Do not use the old initial-observation-first evaluator for this target.
- Do not transfer the selected rank/degree/map/auxiliary law to another model or horizon.
- Do not tune on the sealed claim observations.
- Do not infer a method ranking from SGQF/GenUT gaps or one-seed ESS/runtime.
- Do not describe the route as exact, source-faithful, posterior-valid, HMC-ready, or default-ready.

## Next Justified Phase

If the user authorizes further research, create a fresh tuning scope and output
root for multiple seeds and/or larger `N`, with uncertainty analysis and a
predeclared promotion/veto contract. Preserve CPU-pinned branch preparation
and shared-GPU memory growth. A separate route-governance decision is required
before any leaderboard or production wiring.
