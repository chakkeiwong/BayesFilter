# GenUT Math and Implementation Audit for SV and LGSSM

Date: 2026-07-22

## Objective

Audit the LaTeX derivations and the current code base for the Generalized Unscented Transformation (GenUT) used in the SV and LGSSM paths. The goal is to determine:

1. whether the GenUT math matches the implemented finite computation;
2. whether the implementation is correct for the executed SV and LGSSM routes;
3. whether the current GenUT construction is suitable for other nonlinear models in this repository; and
4. what the main failure modes and transfer limits are.

## What I did

I inspected the following source and artifact groups:

- LaTeX derivations in [docs/chapters/ch32c_entropic_ot_sinkhorn.tex](../../docs/chapters/ch32c_entropic_ot_sinkhorn.tex), especially the GenUT candidate, positivity, and differentiation-contract sections;
- the reusable GenUT design and route-identity code in [bayesfilter/highdim/cubature_genut_candidate.py](../../bayesfilter/highdim/cubature_genut_candidate.py);
- the fixed GenUT adapters in [bayesfilter/highdim/cubature_genut_adapters.py](../../bayesfilter/highdim/cubature_genut_adapters.py);
- the finite value/score core in [bayesfilter/highdim/cubature_genut_filter.py](../../bayesfilter/highdim/cubature_genut_filter.py);
- the LGSSM benchmark and matched-comparison runners in [docs/benchmarks/run_lgssm_cubature_genut_fp32.py](../../docs/benchmarks/run_lgssm_cubature_genut_fp32.py) and [docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py](../../docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py);
- the SV paired GenUT benchmark in [docs/benchmarks/run_exact_sv_fixed_gaussian_genut_paired.py](../../docs/benchmarks/run_exact_sv_fixed_gaussian_genut_paired.py) and [docs/benchmarks/run_genut_antithetic_lgssm_sv.py](../../docs/benchmarks/run_genut_antithetic_lgssm_sv.py);
- the relevant tests in [tests/test_lgssm_cubature_genut_fp32.py](../../tests/test_lgssm_cubature_genut_fp32.py), [tests/highdim/test_cubature_genut_candidate.py](../../tests/highdim/test_cubature_genut_candidate.py), and [tests/highdim/test_genut_antithetic_lgssm_sv.py](../../tests/highdim/test_genut_antithetic_lgssm_sv.py);
- the published result artifacts for the LGSSM and exact-SV campaigns.

## Findings

### 1. The GenUT math is correct for the fixed finite routes that are actually executed

**Classification:** correct

The LaTeX chapter states the right boundary conditions:

- GenUT matches mean/covariance and selected diagonal higher moments in a whitened coordinate system;
- the rule is only a positive OT marginal if its weights are nonnegative;
- exact equal-weight replication is only possible when the weights are exactly representable at the chosen particle count;
- the derivative claim is the total derivative of the **executed finite scalar** on a fixed branch, not the exact posterior score.

That matches the code structure in the current finite routes.

### 2. LGSSM with Gaussian GenUT is not independent evidence from Cubature

**Classification:** wrong relative to the stated target if it is presented as a distinct method result

In the LGSSM path, `STATE_DIM = 3`. The Gaussian GenUT specialization therefore becomes the six-point spherical-radial cubature design exactly:

- the Gaussian GenUT points are `±sqrt(3) e_a`;
- the central weight is zero when `d = 3`;
- the equal-weight replicated design is bitwise identical to Cubature.

This is implemented explicitly in [docs/benchmarks/run_lgssm_cubature_genut_fp32.py:124-139](../../docs/benchmarks/run_lgssm_cubature_genut_fp32.py#L124-L139) and enforced by the LGSSM comparison runner. So the LGSSM “Cubature vs GenUT” comparison is not a real method comparison; it is the same design under two labels.

### 3. The SV fixed-Gaussian-GenUT route is mathematically coherent, but the evidence is narrow

**Classification:** correct for the finite route; unsupported as a general population claim

For the exact transformed SV adapter, the Gaussian GenUT design is a genuine three-point rule in one dimension with weights `2/3, 1/6, 1/6`, and the code differentiates the fixed design as a parameter-independent object. The exact-SV result artifact reports a positive signal on one fresh DGP sequence after the earlier non-DGP arm was revoked.

That is useful evidence for this specific SV diagnostic, but it is not a population proof and it does not establish a general score-bias repair.

### 4. The implementation is a fixed-design score path, not an adaptive GenUT path

**Classification:** correct, but important transfer limit

The current GenUT code path is fixed and parameter-independent:

- `gaussian_genut_design()` constructs a static design;
- `replicate_positive_genut()` turns that into exact equal-weight rows when possible;
- the JVP for the design is effectively zero because the design does not adapt to the current filtering cloud;
- the score path therefore differentiates the executed finite scalar, not a learned or moment-estimated GenUT update.

This is fine for a fixed-design experiment. It is **not** enough for an adaptive GenUT claim.

### 5. The current GenUT construction is only suitable for some nonlinear models in the repo

**Classification:** partly suitable, partly blocked

The main transfer limits are:

- **Dimension limit:** the Gaussian specialization has central weight `1 - d/3`. For `d > 3`, that weight becomes negative, so the rule is not a positive OT marginal and the current replication code rejects it.
- **Representation limit:** even when the weights are positive, exact equal-weight replication requires the particle count to match the rational weights exactly. The design is not universal for arbitrary `N`.
- **Correlation limit:** the LaTeX derivation only matches selected diagonal higher moments in a chosen square-root gauge. It does not match all mixed higher moments of a correlated nonlinear posterior.
- **Adaptivity limit:** the present code is fixed-design only; it does not estimate or differentiate a cloud-adaptive GenUT rule.

So the method is reusable as a **fixed positive residual design** in some low-dimensional models, but it is not a drop-in universal cure for score bias in arbitrary nonlinear state-space models.

## Mathematical audit

### Cubature moments

The cubature design `sqrt(d)[+e_a, -e_a]` with replication has zero mean and identity population covariance under the code's `1/N` convention. This is correct and tested.

### Gaussian GenUT specialization

The chapter's Gaussian specialization `s = 0, k = 3` is correct. In dimension three it collapses to Cubature; in dimension one it becomes the familiar three-point rule; in higher dimensions the central weight becomes smaller and eventually negative.

### Sinkhorn and reset contracts

The finite Sinkhorn route, barycentric map, and Contract E reset in the code are consistent with the LaTeX statements:

- finite kernel scaling uses the mean-cost normalization and the fixed denominator floor;
- the JVP differentiates the executed finite unrolled Sinkhorn loop;
- the reset uses the realized transported covariance, not just the residual design covariance;
- the ridged covariance identity is the one actually proved by the code.

### Score recursion

The LGSSM recursive score path and the SV fixed GenUT path both differentiate the same finite scalar that the value path computes. The score is therefore a correct total derivative of the executed branch, not an exact posterior score.

## Implementation audit

### What looks sound

- The LGSSM value and score recurrences are internally consistent.
- The SV adapter uses the correct transformed-SV equations for the fixed Gaussian GenUT route.
- The reusable GenUT candidate module correctly rejects signed or nonrepresentable weights when they cannot be used as positive OT masses.
- The tests confirm the main fixed-route identities, including cubature moments, Gaussian-GenUT equality to cubature in 3D, and recursive score consistency on small fixtures.

### What is fragile or easy to overclaim

- The LGSSM “GenUT” arm is not distinct from Cubature in 3D, so it should not be advertised as evidence that GenUT fixed the LGSSM score bias.
- The current SV result is single-DGP nomination evidence, not a broad claim about all SV datasets or all nonlinear models.
- An adaptive GenUT would require a different implementation contract because the current code treats the design as fixed and parameter-independent.

## Suitability for other nonlinear models

My assessment is:

- **Suitable as a fixed low-dimensional design:** yes, for models where the Gaussian GenUT rule is positive and exactly representable at the chosen `N`.
- **Suitable for arbitrary nonlinear models:** no, not as currently implemented.
- **Suitable for higher-dimensional nonlinear models:** only if the design is changed or generalized, because the current Gaussian specialization becomes signed for `d > 3` and the present OT machinery requires positive masses.
- **Suitable as an adaptive score-bias repair:** not yet. The code does not currently carry a design tangent for an adaptive moment-estimated GenUT rule.

## Verdict

**VERDICT: REVISE**

The core finite-route mathematics is mostly correct, but the current repo should not present Gaussian GenUT as a universal or LGSSM-distinct cure for score bias.

The biggest issues are:

1. the LGSSM 3D Gaussian GenUT arm is identical to Cubature;
2. the SV evidence is promising but narrow and not population-level;
3. the fixed GenUT route is not an adaptive general nonlinear-model solution;
4. the current positive-mass implementation only covers a subset of dimensions and particle-count choices.

## Suggested next checks for the other agent

1. Keep the LGSSM result labeled as a cubature alias, not as an independent GenUT arm.
2. If the goal is broader nonlinear-model support, decide whether the next step is a fixed-design extension, an adaptive GenUT design, or a different moment-matching route.
3. Add a small matrix of model-dimension checks showing exactly where Gaussian GenUT remains positive, exactly representable, and distinct from Cubature.
4. If score bias is the scientific target, compare the raw physical score error directly, not only any transformed or HMC-scaled relative metric.
