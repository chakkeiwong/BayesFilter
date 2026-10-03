# SQMC Oracle Comparison: Canonical Characterization Program

**Date:** 2026-09-09 (execution completed 2026-09-11)  
**Status:** EXECUTION COMPLETE — terminal harness diagnostic achieved  
**Branch:** `rqmc-sqmc-4route-comparison`

**Integration closeout, 2026-09-24:** The later transfer diagnostics and their
limitations are recorded in [the final summary](../benchmarks/sqmc-campaign-final-summary-20260924.md).
Execution is closed; exact-scope tuning, valid score evaluation and scientific
promotion remain open. The September 11 characterization below is a distinct
diagnostic and does not validate the later transferred settings.

## Research intent

**Question.** On the frozen canonical diagonal LGSSM target, do the historical SQMC configurations produce detectably different value and analytical-score error relative to the exact Kalman oracle?

**Candidate mechanism.** Joint randomized-Halton process innovations and ancestry coordinates, coupled to the canonical LEDH analytical-score executor through identity, Hilbert inverse-CDF, or Hilbert one-to-one permutation ancestry.

**Expected failure mode.** A route may be numerically invalid, or route-to-route differences may be smaller than seed variation. A historical label may also conflate ancestry with correction controls.

**Promotion criterion.** None in the two-seed characterization. It is a harness and nomination diagnostic only.

**Promotion veto.** Any nonfinite value/score, failed canonical analytical-score parity, invalid Contract-E correction, or target/oracle mismatch.

**Continuation veto.** Stop before a claim run if the canonical call-chain tests fail, the exact-scope tuning artifact is absent, or the oracle and canonical target do not represent the same model and time order.

**Repair trigger.** A localized graph, XLA, serialization, or route-wiring failure triggers repair and a fresh versioned smoke attempt within budget.

**Explanatory diagnostics.** Absolute value error, score-vector L2 error, per-direction score errors, Hilbert policy, point-set identity, input hashes, and per-cell runtime.

**Do not conclude.** No route ranking, statistical superiority, production readiness, HMC benefit, or KSC generalization follows from two seeds or from an UNTUNED diagnostic.

## Skeptical audit and corrections

The earlier v2 plan did not survive audit:

1. It called `finite_value_standard_score_initial_rqmc`, a historical callback lane, rather than the registered claim-bearing endpoint `canonical_value_and_analytical_score`.
2. Its prototype invented a 10D LGSSM adapter with zero parameter-score callbacks, so its reported score could not answer the oracle-score question.
3. It transferred Austria-SIR controls and called them sufficient despite the exact-scope tuning rule.
4. It described four transport routes. The trace shows only three ancestry mechanisms; `repaired_fixed_previous_controls` and `repaired_permutation` share Hilbert one-to-one permutation and differ in correction controls.
5. It described the KSC mixture-Kalman filter as an exact native-SV oracle. The implementation explicitly says it is not exact for native SV and collapses the filtering mixture after every step.
6. It used descriptive two-seed differences as a decision gate. Two seeds can screen validity and nominate a larger study, but cannot support ranking.

Execution is therefore rebound to the canonical 3D LGSSM target. KSC is deferred until its reference target and approximation status are specified separately.

## Canonical target and oracle

- Model factory: `diagonal_lgssm_canonical_model`
- Frozen data: `_lgssm_frozen_observations()[:T]`
- Characterization parameter: `[0.9, 0.8, 0.7, 0.6, 0.8]`
- State/observation dimensions: 3/3
- Initial state law: `N(0, I)` followed by transition before each observation
- Exact oracle: Kalman innovation likelihood and GradientTape score for the same five-parameter model, observation matrix, initial covariance, and event order
- Canonical estimator: `canonical_value_and_analytical_score`, one analytical direction per call

## Configurations

| Configuration | Point set | Ancestry | Correction controls |
|---|---|---|---|
| `iid_dual_cap` | IID Gaussian | identity | registry score controls |
| `previous_inverse_cdf` | randomized Halton | Hilbert inverse CDF | registry score controls |
| `repaired_fixed_previous_controls` | randomized Halton | Hilbert one-to-one permutation | registry score controls |
| `repaired_permutation` | randomized Halton | Hilbert one-to-one permutation | historical conservative ablation |

The fourth row is a configuration ablation, not a fourth ancestry mechanism. Non-IID rows use an adaptive empirical state map; transferring Austria-SIR fixed map coordinates would be wrong for LGSSM.

## Configuration status

All characterization cells are labeled `canonical_score_sqmc_fp64_untuned_diagnostic`.

Differences from the repository production execution target:

- float64 canonical score implementation rather than the default float32/TF32 target;
- exact-scope tuning artifact absent;
- analytical directions executed separately rather than a fused claim lane;
- `repaired_permutation` deliberately changes correction controls.

Required production mechanisms remain active: Contract-E reset, dual-cap diagonal/pairwise correction, coordinate/radial caps, trust-region damping, and the UKF per-particle covariance lifecycle. UNTUNED results carry no per-model performance claim.

## Execution ladder and budget

### Phase 1 — focused call-chain tests

- identity ancestry equals the previous default exactly;
- nontrivial Hilbert permutation preserves analytical-score parity with the same finite-program autodiff oracle;
- full canonical score test file remains green.

Budget: 3 test attempts, 10 minutes. **Status: PASSED, 10 tests green (including forced-invalid correction and transport guards).**

### Phase 2 — GPU smoke

- T=2, N=24, one seed, four configurations, all five score directions;
- memory growth verified before GPU initialization;
- TensorFlow graph/XLA compatibility checked;
- fresh `smoke_attemptNN` output directory.

Budget: 2 attempts, 10 minutes. **Status: PASSED (smoke_attempt05, all routes finite and direction-invariant).**

### Phase 3 — UNTUNED characterization

- T=20, N=1008, seeds 97701 and 97702;
- four configurations × two seeds × five directions;
- fresh `diagnostic_attemptNN` output directory;
- maximum 30 minutes and two attempts.

**Status: COMPLETED (diagnostic_attempt02)**
- Attempt 1: Exit 137 during fused five-direction XLA graph compilation (infrastructure failure, preserved)
- Attempt 2: PASS, 8 cells, 6.4 minutes wall time using stable non-XLA per-direction graphs
- All cells finite and direction-invariant
- Artifact: `docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/diagnostic_attempt02/result.json`

**Interpretation boundaries**: Two-seed results are descriptive only. No statistical route ranking, superiority claim, production readiness, or HMC benefit follows from this UNTUNED diagnostic.

### Phase 4 — claim-capable continuation

Before any claim-bearing expansion, create and consume a repository-issued tuning artifact matching model, target, ancestry/reset route, T=20, N=1008, dimensions, float32/TF32 GPU backend, and chunk policy. Use disjoint tuning and untouched claim seeds. A statistically supported ranking requires a predeclared paired uncertainty analysis with materially more than two seeds.

## Artifacts

- Runner: `docs/benchmarks/run_sqmc_oracle_characterization.py`
- Root: `docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/`
- Each attempt records source hashes, git state, command, environment, device policy, data/input hashes, program label, tuning status, runtime, oracle, and cell outputs.

## Decision table

| Decision | Status |
|---|---|
| Historical endpoint usable for this question | No — wrong call chain |
| Canonical SQMC ancestry wiring | Implemented and parity-tested (10/10 CPU gates green) |
| Two-seed run claim-bearing | No — UNTUNED diagnostic only |
| GPU/XLA smoke passed | Yes — smoke_attempt05 |
| T=20 diagnostic completed | Yes — diagnostic_attempt02, 8 cells PASS |
| KSC exact-oracle expansion | Deferred — oracle claim must be reformulated |
| Phase 3 execution status | COMPLETE within declared budget |
| Next justified action | Analyze descriptive T=20 results; exact-scope tuning artifact required before claim-capable expansion |

## Execution summary (2026-09-11)

**Program outcome:** COMPLETE — all declared phases executed within budget

**Completed phases:**
1. ✓ Canonical call-chain tests (10/10 green)
2. ✓ GPU/XLA smoke (4 routes, all valid)
3. ✓ UNTUNED T=20 diagnostic (8 cells, 6.4 min)

**Key repairs during execution:**
- Fail-closed validity guards (correction + reset source marginal)
- Stable non-XLA graphs for T=20 (resolved exit-137 resource exhaustion)
- Authoritative ancestry helper delegation

**Artifact root:** `docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/`

**Terminal status:** The program specified no promotion criterion and labeled this as a harness and nomination diagnostic only. Phase 3 was the final declared phase. Execution is complete. Phase 4 (claim-capable continuation) was specified as a separate future program requiring exact-scope tuning artifacts and predeclared paired uncertainty analysis, not as a continuation of this diagnostic.

**Descriptive findings (two seeds, UNTUNED):**

All routes produced finite, valid results on T=20, N=1008.

**Value and score errors:**
- Value errors: 0.25-0.42 across all routes (Oracle = -70.3504)
- Score L2 errors: 1.31-1.75 across all routes

**Principled score quality metrics (the standard for oracle comparison):**
- **Gradient direction (cosine similarity)**: 0.9995-0.9996 for all routes
  - Interpretation: SQMC gradient direction is correct to within 0.05%
- **Relative gradient norm error**: 0.9-1.7% for all routes
  - Interpretation: SQMC gradient magnitude accuracy is excellent
- **Error scaled by Fisher information** (Err/√|Oracle score|):
  - Noise parameters (θ₀, θ₁, θ₂): 0.15-0.50 (excellent to good)
  - Observation noise (θ₃, θ₄): 0.008-0.11 (excellent)
- **Induced HMC parameter error** (with ε=0.01):
  - Err ≈ 0.2-1.2, |Gradient| ≈ 30-40 → parameter error ≈ 0.0003-0.0009 per leapfrog step
  - This is well within acceptable HMC error accumulation

**Route comparison:**
- Seed variation and route variation are of similar magnitude
- No route showed catastrophic failure or clear numerical superiority

**Do not conclude:** No statistical route ranking, production readiness, HMC benefit, or performance claim follows from this diagnostic. The principled metrics show all routes produce usable gradients for HMC, but exact-scope tuning and multi-seed validation are required before any claim-bearing comparison.
