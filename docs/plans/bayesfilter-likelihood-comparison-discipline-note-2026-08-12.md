# Likelihood-comparison discipline for statistical approximation studies

Date: 2026-08-12
Status: `ACTIVE_COMPARISON_DISCIPLINE`

## Why this note exists

Recent actual-SV comparison work exposed a recurring failure mode:
we drifted from a statistical comparison question into an implementation-path comparison question, and that produced misleading conclusions.

The core lesson is simple:
for nonlinear filtering models, we must compare **approximations to the same likelihood function** in a mathematically aligned representation before interpreting gaps as bugs or model failures.

This note records the required discipline for future model × algorithm comparisons in this repo.

## Governing statistical principle

For a model with parameter `theta` and observed data `y_{0:T}`, the exact likelihood function is

\[
L(\theta) = p_\theta(y_{0:T}).
\]

This object is unique.

Approximation methods do **not** redefine the likelihood. They define approximate evaluations

\[
\hat L_M(\theta) \approx L(\theta),
\]

where `M` is the approximation budget / family choice.

If two routes are being compared as approximations to the same model, then they must both be interpreted as approximations to the same exact `L(theta)` after any exact change of variables is handled correctly.

## Required comparison protocol

### Step 1 — State the exact target first
Before discussing any algorithm, write the exact statistical target:
- the model equations,
- the exact likelihood `L(theta)`,
- and, if a transformation is used, the exact transformed likelihood relation.

Example:
if `z = g(y)` is used, record explicitly how

\[
\ell_Y(\theta) = \ell_Z(\theta) + J(y)
\]

with `J(y)` data-only if applicable.

### Step 2 — Name each approximation family explicitly
For every algorithm, record:
- what approximation family it belongs to,
- what budget parameter it uses,
- and whether it approximates:
  - the exact transformed target,
  - a Gaussian-closure surrogate,
  - a finite Gaussian-mixture surrogate,
  - or some other declared approximate model.

Never use vague wording like “same scalar” without naming the underlying statistical target.

### Step 3 — Separate exact transformations from approximations
An exact transformation (for example `z = log(y^2)` with the exact Jacobian correction) is not itself an approximation.

But two routes that both use the transformed model can still be **different approximation families**.

So every comparison must say explicitly:
- what part is exact mathematics,
- what part is approximation.

### Step 4 — Always do within-family validation before cross-family comparison
Before comparing method A to method B, first validate each method against its own closest same-target benchmark.

Required order:
1. route vs its own dense / same-target reference,
2. route score vs finite differences of its own value,
3. only then route vs route.

Interpretation rule:
- if a route fails against its own same-target benchmark, do not use it in cross-family conclusions.

### Step 5 — Put all routes into the same likelihood representation before comparison
If one route works in transformed-observation space and another is interpreted in raw-observation space, first map both into a common representation using the exact change-of-variables relation.

Do not compare raw and transformed likelihood numbers directly unless the exact transformation correction has already been applied.

### Step 6 — Distinguish four explanations for a gap
When two routes differ, treat the live explanations as:
1. exact transformation mismatch,
2. finite approximation-budget error,
3. approximation-family difference,
4. bug.

Do not jump directly from “gap exists” to “bug”.

### Step 7 — Use refinement ladders before bug claims
For any approximation family with a natural budget ladder (mixture size, quadrature order, SGQF level, TT rank/degree, particle count), run refinement before declaring a route broken.

Interpretation rule:
- if refinement materially changes the answer, the earlier budget was not sufficient;
- if refinement stabilizes quickly, residual disagreement is more likely due to approximation-family difference or a bug.

### Step 8 — Use simulation for arbitration when needed
If within-family checks pass but cross-family disagreement remains, simulate from the exact model and compare both approximations on the same simulated datasets.

This is the preferred arbitration step when deciding whether the remaining issue is on route A, route B, or neither.

## Language rules for future discussions

Use statistical / economic language first:
- exact likelihood,
- approximation family,
- finite-budget adequacy,
- convergence under refinement,
- transformed target,
- closure approximation,
- score as derivative of the approximate likelihood.

Avoid leading with computer-science language like:
- backend,
- route wiring,
- same scalar,
- code path,
unless the issue is truly implementation-specific.

## Required checklist before accepting a comparison result

Before calling any comparison result trustworthy, verify all of the following:
- [ ] exact target likelihood has been written down;
- [ ] exact transformation relation is explicit;
- [ ] each route’s approximation family is named;
- [ ] within-family validation succeeded for each route used in the comparison;
- [ ] all compared likelihoods are in the same mathematical representation;
- [ ] at least one refinement ladder has been checked where applicable;
- [ ] if disagreement remains, simulation-based arbitration has been considered.

## What went wrong in the actual-SV discussion

The earlier misleading conclusion came from comparing different approximation constructions before first aligning them as approximations to the same likelihood in the same representation.

The later corrected comparison — transformed back to the same raw-`y` likelihood and checked under refinement — showed that the apparent large discrepancy was mainly a comparison-design error.

This specific mistake must not be repeated for the remaining models and comparison algorithms.

## Recommended next use

Use this note as a mandatory preamble for future work on:
- remaining stochastic-volatility model comparisons,
- dense-reference vs approximate-filter comparisons,
- score-comparison studies,
- any cross-family benchmark where transformed and raw representations can be mixed.
